"""Model fitting and explanations on a synthetic problem with a known driver."""

import numpy as np
import pandas as pd

from mortfair import explain as EX
from mortfair import models as M
from mortfair.features import CATEGORICAL, FEATURES, GROUP_OF


def _data(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))).astype(np.float32), columns=FEATURES)
    for c in CATEGORICAL:
        X[c] = rng.integers(0, 3, n).astype(np.float32)
    X["dti"] = rng.uniform(10, 65, n).astype(np.float32)
    X["dti_missing"] = 0.0
    logit = -4 + 0.12 * (X.dti - 30)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y.to_numpy()


def test_tuning_shap_additivity_and_top_feature() -> None:
    Xtr, ytr = _data(3000, 0)
    Xva, yva = _data(1500, 1)
    t = M.tune_lightgbm(Xtr, ytr, Xva, yva, n_trials=2)
    assert len(t.trials) == 2 and t.val_log_loss < M.log_loss(yva, np.full(len(yva), ytr.mean()))
    sv, base = M.tree_shap(t.model, Xva.iloc[:200], t.best_iteration)
    raw = t.model.predict(Xva.iloc[:200], num_iteration=t.best_iteration, raw_score=True)
    assert np.allclose(sv.sum(1) + base, raw, atol=1e-6)
    assert FEATURES[int(np.argmax(np.abs(sv).mean(0)))] == "dti"
    lookup = M.DtiLookup().fit(Xtr, ytr)
    assert lookup.predict_proba(Xva).shape == (len(Xva), 2)


def test_lime_finds_the_driver_and_agreement_metrics() -> None:
    Xtr, ytr = _data(3000, 2)
    lr = M.logistic_pipeline(1.0, {c: 3 for c in CATEGORICAL}).fit(Xtr, ytr)
    rows = Xtr.iloc[:3].to_numpy(np.float64)
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
    w = EX.lime_attributions(lambda A: lr.predict_proba(pd.DataFrame(A, columns=FEATURES)),
                             Xtr.to_numpy(np.float64), rows, FEATURES, cat_idx, seed=0, num_samples=1500)
    assert (np.argmax(np.abs(w), axis=1) == FEATURES.index("dti")).all()
    a = np.array([3.0, -2.0, 1.0, 0.1, 0.0])
    same = EX.agreement(a, a, k=3)
    assert same["top_k_overlap"] == 1 and same["sign_agreement"] == 1 and np.isclose(same["rank_corr_abs"], 1)
    flipped = EX.agreement(a, np.array([0.0, 0.1, 0.2, 5.0, 4.0]), k=2)
    assert flipped["top_k_overlap"] == 0 and flipped["same_top_feature"] == 0


def test_reason_match_and_feature_groups() -> None:
    assert set(GROUP_OF) == set(FEATURES)
    attr = np.zeros((4, len(FEATURES)))
    attr[0, FEATURES.index("dti")] = 2.0
    attr[1, FEATURES.index("cltv")] = 2.0
    attr[2, FEATURES.index("dti")] = 1.0
    attr[3, FEATURES.index("lender")] = 1.0
    top = EX.top_denial_group(attr, FEATURES)
    reason = pd.Series(["Debt-to-income ratio", "Collateral", "Collateral", "Debt-to-income ratio"])
    rm = EX.reason_match(top, reason).set_index("reported_reason")
    assert rm.loc["Debt-to-income ratio", "match_rate"] == 0.5 and rm.loc["Collateral", "match_rate"] == 0.5
