"""Synthetic HMDA-like records for the tests (same column names and codes as the Data Browser CSV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mortfair.data import FILTER_STEPS, NOT_FLAGS

TRACT_COLS = ["tract_to_msa_income_percentage", "tract_median_age_of_housing_units", "tract_one_to_four_family_homes",
              "tract_owner_occupied_units", "ffiec_msa_md_median_family_income", "tract_minority_population_percent"]


def make_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Records that pass the population filter, with DTI driving denial and a few 'NA' codes."""
    rng = np.random.default_rng(seed)
    dti_codes = np.array(["<20%", "20%-<30%", "30%-<36%", "40", "45", "49", "50%-60%", ">60%", "NA"])
    dti = rng.choice(dti_codes, n, p=[0.1, 0.2, 0.2, 0.15, 0.1, 0.1, 0.07, 0.05, 0.03])
    risk = np.select([dti == ">60%", dti == "50%-60%", dti == "NA"], [0.9, 0.6, 0.4], 0.04)
    denied = rng.random(n) < risk
    df = pd.DataFrame({col: [keep[0]] * n for _, col, keep in FILTER_STEPS})
    df["action_taken"] = np.where(denied, "3", "1")
    for c in NOT_FLAGS:
        df[c] = "2"
    df["loan_amount"] = (rng.integers(10, 80, n) * 10000 + 5000).astype(str)
    df["property_value"] = (rng.integers(30, 120, n) * 10000 + 5000).astype(str)
    df["loan_to_value_ratio"] = rng.uniform(40, 100, n).round(2).astype(str)
    df["income"] = rng.integers(40, 300, n).astype(str)
    df["debt_to_income_ratio"] = dti
    df["loan_term"] = "360"
    df["intro_rate_period"] = "NA"
    df["census_tract"] = "04013000100"
    for c in TRACT_COLS:
        df[c] = "100"
    for c in ["interest_only_payment", "balloon_payment", "negative_amortization", "other_nonamortizing_features"]:
        df[c] = "2"
    df["conforming_loan_limit"] = "C"
    df["submission_of_application"] = "1"
    df["co-applicant_age"] = rng.choice(["9999", "35-44"], n)
    df["lei"] = rng.choice(["LENDER_A", "LENDER_B", "LENDER_C"], n)
    df["county_code"] = rng.choice(["04013", "04019"], n)
    df["aus-1"] = "1"
    df["applicant_credit_score_type"] = "1"
    df["derived_race"] = rng.choice(["White", "Black or African American", "Asian", "Race Not Available"], n)
    df["derived_ethnicity"] = rng.choice(["Not Hispanic or Latino", "Hispanic or Latino"], n)
    df["derived_sex"] = rng.choice(["Male", "Female", "Joint"], n)
    df["applicant_age"] = rng.choice(["25-34", "35-44", "65-74", "8888"], n)
    df["applicant_age_above_62"] = np.where(df.applicant_age == "65-74", "Yes", "No")
    for i in range(1, 5):
        df[f"denial_reason-{i}"] = np.where(denied & (i == 1), "1", np.where(denied, "", "10"))
    return df


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()
