"""Typed application table, protected attributes (audit only), excluded fields and the model feature encoder.

HMDA codes 'NA' (not applicable / not reported) become NaN with an explicit missing-value flag; partially exempt
records ('Exempt' / 1111) are removed by the population filter before this module sees them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Debt-to-income is published in bins below 36% and above 49%, and as whole percentages from 36 to 49.
DTI_BINS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}

DENIAL_REASONS = {"1": "Debt-to-income ratio", "2": "Employment history", "3": "Credit history", "4": "Collateral",
                  "5": "Insufficient cash", "6": "Unverifiable information", "7": "Application incomplete",
                  "8": "Mortgage insurance denied", "9": "Other"}

RACE_MAP = {"White": "White", "Black or African American": "Black", "Asian": "Asian",
            "American Indian or Alaska Native": "American Indian / Alaska Native",
            "Native Hawaiian or Other Pacific Islander": "Other minority races",
            "2 or more minority races": "Other minority races", "Free Form Text Only": "Other minority races",
            "Joint": "Joint (two races)", "Race Not Available": "Not available"}
ETHNICITY_MAP = {"Hispanic or Latino": "Hispanic or Latino", "Not Hispanic or Latino": "Not Hispanic or Latino",
                 "Joint": "Joint (Hispanic and not)", "Ethnicity Not Available": "Not available",
                 "Free Form Text Only": "Not available"}
SEX_MAP = {"Male": "Male", "Female": "Female", "Joint": "Joint (male and female)", "Sex Not Available": "Not available"}
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]

ATTRIBUTES = ("race", "ethnicity", "sex", "age")
REFERENCE = {"race": "White", "ethnicity": "Not Hispanic or Latino", "sex": "Male", "age": "35-44"}

NUMERIC = ["loan_amount_k", "property_value_k", "cltv", "income_k", "loan_to_income", "dti", "loan_term",
           "intro_rate_months", "tract_income_pct", "tract_housing_age", "tract_owner_share", "msa_income_k"]
BINARY = ["jumbo", "interest_only", "balloon", "negative_amortization", "other_nonamortizing", "submitted_direct",
          "has_coapplicant"]
FLAGS = ["property_value_missing", "cltv_missing", "income_missing", "dti_missing", "loan_term_missing",
         "location_missing"]
CATEGORICAL = ["lender", "county", "aus", "credit_model"]
FEATURES = NUMERIC + BINARY + FLAGS + CATEGORICAL

GROUPS = {
    "Debt-to-income": ["dti"],
    "Collateral (LTV, value)": ["cltv", "property_value_k"],
    "Income and loan size": ["income_k", "loan_to_income", "loan_amount_k", "jumbo"],
    "Missing fields": FLAGS,
    "Lender": ["lender"],
    "Underwriting channel": ["aus", "credit_model", "submitted_direct"],
    "Loan terms": ["loan_term", "intro_rate_months", "interest_only", "balloon", "negative_amortization",
                   "other_nonamortizing"],
    "Location": ["county", "tract_income_pct", "tract_housing_age", "tract_owner_share", "msa_income_k"],
    "Co-applicant": ["has_coapplicant"],
}
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}

LABELS = {"loan_amount_k": "Loan amount ($k)", "property_value_k": "Property value ($k)", "cltv": "Combined LTV (%)",
          "income_k": "Income ($k)", "loan_to_income": "Loan / income", "dti": "Debt-to-income (%)",
          "loan_term": "Loan term (months)", "intro_rate_months": "Intro-rate period (months)",
          "tract_income_pct": "Tract income / MSA (%)", "tract_housing_age": "Tract housing age (years)",
          "tract_owner_share": "Tract owner-occupied share", "msa_income_k": "MSA median family income ($k)",
          "jumbo": "Jumbo (non-conforming)", "interest_only": "Interest-only", "balloon": "Balloon payment",
          "negative_amortization": "Negative amortization", "other_nonamortizing": "Other non-amortizing",
          "submitted_direct": "Applied directly to lender", "has_coapplicant": "Has co-applicant",
          "property_value_missing": "Property value missing", "cltv_missing": "CLTV missing",
          "income_missing": "Income missing", "dti_missing": "DTI missing", "loan_term_missing": "Loan term missing",
          "location_missing": "Census tract missing", "lender": "Lender", "county": "County",
          "aus": "Automated underwriting system", "credit_model": "Credit score model"}

# Fields that are not model inputs, and why. Protected attributes and the tract minority share are kept for the audit.
EXCLUDED = {
    "derived_race / applicant & co-applicant race": "protected attribute (audit only)",
    "derived_ethnicity / applicant & co-applicant ethnicity": "protected attribute (audit only)",
    "derived_sex / applicant & co-applicant sex": "protected attribute (audit only)",
    "applicant_age, co-applicant_age, *_age_above_62": "protected attribute (audit only)",
    "*_observed (race/ethnicity/sex collected by visual observation)": "protected-attribute metadata",
    "tract_minority_population_percent": "close geographic proxy for race (audit only)",
    "interest_rate, rate_spread, total_loan_costs, origination_charges, discount_points, lender_credits, "
    "total_points_and_fees": "pricing fields, only reported for originated loans (outcome leak)",
    "hoepa_status": "'not applicable' (3) for every denial (outcome leak)",
    "purchaser_type": "0 (not sold) for every denial (outcome leak)",
    "denial_reason-1..4": "reported only for denials; used to validate explanations, never as a feature",
    "preapproval": "denied pre-approval requests are action 7, outside this population, so 'requested' "
                   "implies approval by construction",
    "initially_payable_to_institution": "code 3 'not applicable' occurs only for denials (outcome leak)",
    "co-applicant_credit_score_type": "duplicates has_coapplicant; otherwise not informative",
    "prepayment_penalty_term, multifamily_affordable_units, manufactured_home_*": "constant or not applicable "
                                                                                   "in this population",
    "derived_msa-md, state_code, census_tract": "represented by county and tract-level features",
}


def to_number(s: pd.Series) -> pd.Series:
    """HMDA text column to float; 'NA', 'Exempt' and blanks become NaN."""
    return pd.to_numeric(s.where(~s.isin(["NA", "Exempt", ""])), errors="coerce")


def parse_dti(s: pd.Series) -> pd.Series:
    """Debt-to-income bins and whole percentages to a number (bin midpoints; NaN for NA)."""
    out = s.map(DTI_BINS)
    whole = pd.to_numeric(s, errors="coerce")
    return out.fillna(whole).astype(float)


def _age_group(s: pd.Series) -> pd.Series:
    return s.where(s.isin(AGE_ORDER), "Not available")


def application_table(pop: pd.DataFrame) -> pd.DataFrame:
    """One typed row per application: label, raw model inputs (NaN = not reported), protected attributes, reasons."""
    t = pd.DataFrame(index=pop.index)
    t["denied"] = (pop["action_taken"] == "3").astype(int)
    loan = to_number(pop["loan_amount"])
    t["loan_amount_k"] = loan / 1e3
    t["property_value_k"] = to_number(pop["property_value"]) / 1e3
    t["cltv"] = to_number(pop["loan_to_value_ratio"]).clip(0, 200)
    t["income_k"] = to_number(pop["income"]).clip(-100, 5000)
    t["loan_to_income"] = (t.loan_amount_k / t.income_k.where(t.income_k > 0)).clip(upper=50)
    t["dti"] = parse_dti(pop["debt_to_income_ratio"])
    t["loan_term"] = to_number(pop["loan_term"])
    t["intro_rate_months"] = to_number(pop["intro_rate_period"]).fillna(0.0)  # NA = fixed rate
    located = pop["census_tract"] != "NA"
    t["tract_income_pct"] = to_number(pop["tract_to_msa_income_percentage"]).where(located)
    t["tract_housing_age"] = to_number(pop["tract_median_age_of_housing_units"]).where(located)
    homes = to_number(pop["tract_one_to_four_family_homes"])
    t["tract_owner_share"] = (to_number(pop["tract_owner_occupied_units"]) / homes.where(homes > 0)).clip(0, 1.5)
    t["tract_owner_share"] = t["tract_owner_share"].where(located)
    t["msa_income_k"] = (to_number(pop["ffiec_msa_md_median_family_income"]) / 1e3).where(located)
    t["jumbo"] = (pop["conforming_loan_limit"] == "NC").astype(int)
    for src, dst in [("interest_only_payment", "interest_only"), ("balloon_payment", "balloon"),
                     ("negative_amortization", "negative_amortization"),
                     ("other_nonamortizing_features", "other_nonamortizing")]:
        t[dst] = (pop[src] == "1").astype(int)
    t["submitted_direct"] = (pop["submission_of_application"] == "1").astype(int)
    t["has_coapplicant"] = (pop["co-applicant_age"] != "9999").astype(int)
    t["location_missing"] = (~located).astype(int)
    t["lender_id"] = pop["lei"].to_numpy()
    t["county_code"] = pop["county_code"].to_numpy()
    t["aus_code"] = pop["aus-1"].to_numpy()
    t["credit_model_code"] = pop["applicant_credit_score_type"].to_numpy()
    # protected attributes and proxies (never model inputs)
    t["race"] = pop["derived_race"].map(RACE_MAP)
    t["ethnicity"] = pop["derived_ethnicity"].map(ETHNICITY_MAP)
    t["sex"] = pop["derived_sex"].map(SEX_MAP)
    t["age"] = _age_group(pop["applicant_age"])
    t["age_62_plus"] = (pop["applicant_age_above_62"] == "Yes").astype(int)
    t["tract_minority_pct"] = to_number(pop["tract_minority_population_percent"]).where(located)
    reasons = pop[[f"denial_reason-{i}" for i in range(1, 5)]]
    t["denial_reason_1"] = pop["denial_reason-1"].map(DENIAL_REASONS)
    t["denial_reasons"] = reasons.apply(lambda r: "|".join(DENIAL_REASONS[c] for c in r if c in DENIAL_REASONS), axis=1)
    if t[["race", "ethnicity", "sex"]].isna().any().any():
        raise ValueError("unmapped derived race / ethnicity / sex value")
    return t.reset_index(drop=True)


def missingness_by_outcome(t: pd.DataFrame) -> pd.DataFrame:
    """Share of 'NA' (not reported) per raw field among approved and denied applications."""
    rows = []
    for col in ["property_value_k", "cltv", "income_k", "dti", "loan_term", "tract_income_pct"]:
        na = t[col].isna()
        rows.append({"field": col, "missing_approved": na[t.denied == 0].mean(), "missing_denied": na[t.denied == 1].mean(),
                     "denial_rate_if_missing": t.denied[na].mean() if na.any() else np.nan,
                     "denial_rate_if_present": t.denied[~na].mean(), "n_missing": int(na.sum())})
    return pd.DataFrame(rows)


@dataclass
class Encoder:
    """Turns the application table into a numeric, NaN-free model matrix using statistics of the training rows only.

    Numeric fields are median-imputed with a missing-value flag; categoricals become integer codes (rare levels share
    one 'other' code). The same matrix feeds every model and both explainers.
    """

    top_lenders: int = 40
    min_county_rows: int = 50
    medians: dict[str, float] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, t: pd.DataFrame) -> "Encoder":
        """Learn medians and category vocabularies from training rows."""
        self.medians = {c: float(t[c].median()) for c in NUMERIC}
        self.vocab = {
            "lender": t.lender_id.value_counts().index[: self.top_lenders].tolist(),
            "county": sorted(c for c, n in t.county_code.value_counts().items() if n >= self.min_county_rows and c != "NA"),
            "aus": sorted(c for c, n in t.aus_code.value_counts().items() if n >= 20),
            "credit_model": sorted(c for c, n in t.credit_model_code.value_counts().items() if n >= 20),
        }
        return self

    def n_levels(self, name: str) -> int:
        """Number of integer codes of a categorical feature (vocabulary plus 'other')."""
        return len(self.vocab[name]) + 1

    def level_names(self, name: str) -> list[str]:
        """Readable level names in code order."""
        return [*self.vocab[name], "other"]

    def transform(self, t: pd.DataFrame) -> pd.DataFrame:
        """Model matrix (float32) with columns in FEATURES order."""
        X = pd.DataFrame(index=t.index)
        for c in NUMERIC:
            X[c] = t[c].fillna(self.medians[c])
        for c in BINARY:
            X[c] = t[c]
        X["property_value_missing"] = t.property_value_k.isna().astype(int)
        X["cltv_missing"] = t.cltv.isna().astype(int)
        X["income_missing"] = t.income_k.isna().astype(int)
        X["dti_missing"] = t.dti.isna().astype(int)
        X["loan_term_missing"] = t.loan_term.isna().astype(int)
        X["location_missing"] = t.location_missing
        for name, src in [("lender", "lender_id"), ("county", "county_code"), ("aus", "aus_code"),
                          ("credit_model", "credit_model_code")]:
            codes = {v: i for i, v in enumerate(self.vocab[name])}
            X[name] = t[src].map(codes).fillna(len(codes)).astype(int)
        return X[FEATURES].astype(np.float32)
