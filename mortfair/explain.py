"""Local explanations (TreeSHAP and LIME), their agreement, and a check of explanations against the denial reasons
lenders reported to HMDA."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from mortfair.features import GROUP_OF, GROUPS
from mortfair.metrics import spearman

# Reported denial reason -> feature group that should carry the explanation if the model mirrors that reason.
REASON_TO_GROUP = {"Debt-to-income ratio": "Debt-to-income", "Collateral": "Collateral (LTV, value)",
                   "Insufficient cash": "Collateral (LTV, value)", "Application incomplete": "Missing fields"}


def lime_attributions(predict_proba: Callable[[np.ndarray], np.ndarray], train: np.ndarray, rows: np.ndarray,
                      feature_names: list[str], categorical_idx: list[int], seed: int,
                      num_samples: int = 5000) -> np.ndarray:
    """LIME weights for every feature of every row (quartile discretisation, categorical features sampled from training
    frequencies). A positive weight pushes towards denial."""
    explainer = LimeTabularExplainer(train, mode="classification", feature_names=feature_names,
                                     categorical_features=categorical_idx, discretize_continuous=True,
                                     random_state=seed)
    out = np.zeros((len(rows), len(feature_names)))
    for i, r in enumerate(rows):
        exp = explainer.explain_instance(r, predict_proba, labels=(1,), num_features=len(feature_names),
                                         num_samples=num_samples)
        for j, w in exp.as_map()[1]:
            out[i, j] = w
    return out


def top_k(a: np.ndarray, k: int) -> set[int]:
    """Indices of the k largest absolute attributions."""
    return set(np.argsort(-np.abs(a), kind="stable")[:k].tolist())


def agreement(a: np.ndarray, b: np.ndarray, k: int = 5) -> dict[str, float]:
    """Agreement of two attribution vectors for one case: top-k overlap, sign agreement on the shared top-k features,
    rank correlation of |attribution| over all features, and whether the top feature is the same."""
    ta, tb = top_k(a, k), top_k(b, k)
    shared = sorted(ta & tb)
    return {"top_k_overlap": len(shared) / k,
            "sign_agreement": float(np.mean(np.sign(a[shared]) == np.sign(b[shared]))) if shared else np.nan,
            "rank_corr_abs": spearman(np.abs(a), np.abs(b)),
            "same_top_feature": float(int(np.argmax(np.abs(a))) == int(np.argmax(np.abs(b))))}


def agreement_table(a: np.ndarray, b: np.ndarray, ks: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """Per-case agreement between two attribution matrices (rows = cases) for several k."""
    rows = []
    for i in range(len(a)):
        for k in ks:
            rows.append({"case": i, "k": k, **agreement(a[i], b[i], k)})
    return pd.DataFrame(rows)


def group_attributions(attr: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Sum attributions within each feature group (columns ordered as in GROUPS)."""
    out = pd.DataFrame(0.0, index=range(len(attr)), columns=list(GROUPS))
    for j, f in enumerate(feature_names):
        out[GROUP_OF[f]] += attr[:, j]
    return out


def top_denial_group(attr: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Feature group with the largest push towards denial for each case."""
    return group_attributions(attr, feature_names).idxmax(axis=1)


def reason_match(top_group: pd.Series, reason: pd.Series) -> pd.DataFrame:
    """Share of denials whose top explanation group matches the lender's first reported denial reason, against the
    same share among denials with other reasons (the match rate expected without any link)."""
    rows = []
    for r, g in REASON_TO_GROUP.items():
        has = reason == r
        if has.sum() == 0:
            continue
        hit = top_group == g
        rows.append({"reported_reason": r, "expected_group": g, "n_denials": int(has.sum()),
                     "match_rate": float(hit[has].mean()), "match_rate_other_reasons": float(hit[~has].mean()),
                     "lift": float(hit[has].mean() / max(hit[~has].mean(), 1e-9))})
    return pd.DataFrame(rows)
