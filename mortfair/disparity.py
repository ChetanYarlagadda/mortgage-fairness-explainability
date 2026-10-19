"""Observational denial-gap analysis: odds ratios of denial by group, unadjusted and adjusted for the application
fields HMDA reports (logistic regression with heteroskedasticity-robust standard errors).

Adjusted odds ratios describe association after conditioning on observed fields. HMDA has no credit score, assets
or full credit history, so a remaining gap cannot be attributed to discrimination, nor ruled out as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

DTI_EDGES = [-np.inf, 19.9, 29.9, 35.9, 41.9, 45.9, 49.9, 60.1, np.inf]
DTI_LABELS = ["<20", "20-29", "30-35", "36-41", "42-45", "46-49", "50-60", ">60"]
CLTV_EDGES = [-np.inf, 60, 80, 90, 95, 100, np.inf]
CLTV_LABELS = ["<=60", "60-80", "80-90", "90-95", "95-100", ">100"]


def _bucket(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    b = pd.cut(x, edges, labels=labels).astype(object)
    return b.where(x.notna(), "missing")


def control_frame(t: pd.DataFrame, lenders: bool, min_level_rows: int = 30, min_level_outcomes: int = 5) -> pd.DataFrame:
    """Design matrix of application controls (no protected attributes): DTI and CLTV buckets, log income and loan size,
    product and channel fields, county, and optionally lender dummies.

    Levels with fewer than ``min_level_rows`` rows, or fewer than ``min_level_outcomes`` denials or approvals, are
    pooled into one 'rare' level so that no dummy perfectly separates the outcome."""
    X = pd.DataFrame(index=t.index)
    X["log_income"] = np.log(t.income_k.clip(lower=1)).fillna(np.log(t.income_k.clip(lower=1)).median())
    X["income_missing"] = t.income_k.isna().astype(float)
    X["log_loan"] = np.log(t.loan_amount_k)
    X["log_loan_to_income"] = np.log(t.loan_to_income.clip(lower=0.01)).fillna(
        np.log(t.loan_to_income.clip(lower=0.01)).median())
    X["property_value_missing"] = t.property_value_k.isna().astype(float)
    X["loan_term_missing"] = t.loan_term.isna().astype(float)
    X["term_not_360"] = ((t.loan_term != 360) & t.loan_term.notna()).astype(float)
    X["adjustable_rate"] = (t.intro_rate_months > 0).astype(float)
    tip = t.tract_income_pct
    X["tract_income_z"] = ((tip - tip.mean()) / tip.std()).fillna(0.0)
    for c in ["jumbo", "interest_only", "balloon", "negative_amortization", "other_nonamortizing", "submitted_direct",
              "has_coapplicant", "location_missing"]:
        X[c] = t[c].astype(float)
    cats = {"dti": (_bucket(t.dti, DTI_EDGES, DTI_LABELS), "30-35"),
            "cltv": (_bucket(t.cltv, CLTV_EDGES, CLTV_LABELS), "60-80"),
            "county": (t.county_code, t.county_code.mode()[0]),
            "aus": (t.aus_code, t.aus_code.mode()[0]),
            "credit_model": (t.credit_model_code, t.credit_model_code.mode()[0])}
    if lenders:
        cats["lender"] = (t.lender_id, "rare")
    for name, (s, ref) in cats.items():
        st = t.denied.groupby(s).agg(["sum", "size"])
        ok = st[(st["size"] >= min_level_rows) & (st["sum"] >= min_level_outcomes)
                & (st["size"] - st["sum"] >= min_level_outcomes)].index
        s = s.where(s.isin(ok), "rare")
        d = pd.get_dummies(s, prefix=name, dtype=float)
        drop = f"{name}_{ref}"
        X = pd.concat([X, d.drop(columns=[c for c in d.columns if c == drop])], axis=1)
    return X.loc[:, X.std() > 0]


def odds_ratios(t: pd.DataFrame, attribute: str, reference: str, controls: pd.DataFrame | None,
                min_group_rows: int = 150) -> pd.DataFrame:
    """Denial odds ratio of each group against ``reference`` with 95% CIs (HC1 robust SEs).

    Groups smaller than ``min_group_rows`` are kept in the regression under one 'small groups' level and not reported.
    """
    g = t[attribute].copy()
    sizes = g.value_counts()
    g = g.where(g.map(sizes) >= min_group_rows, "small groups")
    D = pd.get_dummies(g, dtype=float).drop(columns=[reference])
    X = D if controls is None else pd.concat([D, controls], axis=1)
    X = sm.add_constant(X, has_constant="add")
    res = sm.Logit(t.denied.astype(float), X).fit(disp=0, method="newton", maxiter=200, cov_type="HC1")
    rows = []
    for grp in D.columns:
        if grp == "small groups":
            continue
        b, se = float(res.params[grp]), float(res.bse[grp])
        rows.append({"group": grp, "n": int((g == grp).sum()), "odds_ratio": float(np.exp(b)),
                     "ci_low": float(np.exp(b - 1.96 * se)), "ci_high": float(np.exp(b + 1.96 * se)),
                     "p_value": float(res.pvalues[grp]), "n_params": int(X.shape[1]), "converged": bool(res.mle_retvals["converged"])})
    return pd.DataFrame(rows)
