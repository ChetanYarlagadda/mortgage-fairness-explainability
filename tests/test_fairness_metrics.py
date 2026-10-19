"""Fairness audit, reweighing, threshold post-processing, metrics and the odds-ratio analysis on synthetic data."""

import numpy as np
import pandas as pd

from mortfair import fairness as F
from mortfair import metrics as M
from mortfair.disparity import odds_ratios


def _groups(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = rng.choice(["A", "B"], n, p=[0.7, 0.3])
    x = rng.normal(size=n) + (g == "B") * 0.8  # group B has worse scores on average
    p = 1 / (1 + np.exp(-(x * 2 - 2)))
    y = (rng.random(n) < p).astype(int)
    return g, p, y


def test_reweighing_makes_label_independent_of_group() -> None:
    g, _, y = _groups(5000, 0)
    w = F.reweighing_weights(g, y)
    rates = pd.Series(w * y).groupby(g).sum() / pd.Series(w).groupby(g).sum()
    assert abs(rates["A"] - rates["B"]) < 1e-9 and abs(rates["A"] - y.mean()) < 1e-9


def test_group_table_rates_and_calibration() -> None:
    g, p, y = _groups(6000, 1)
    deny = p >= 0.5
    tab = F.group_table(y, p, deny, g, reference="A", n_boot=200)
    b = tab.set_index("group").loc["B"]
    m = g == "B"
    assert np.isclose(b.observed_denial_rate, y[m].mean()) and np.isclose(b.model_approval_rate, 1 - deny[m].mean())
    assert np.isclose(b.true_approval_rate, (~deny[m & (y == 0)]).mean())
    assert abs(b.calibration_gap) < 0.03  # p is the true probability, so the model is calibrated within group
    assert b.observed_denial_rate_diff_ci_low > 0  # group B is denied more often
    tab0 = tab.set_index("group").loc["A"]
    assert tab0.model_approval_rate_diff == 0


def test_threshold_optimizer_narrows_approval_gap() -> None:
    g, p, y = _groups(8000, 2)
    gv, pv, yv = _groups(8000, 3)
    thr = F.accuracy_threshold(yv, pv)
    approve_single = (p < thr).astype(int)
    pp = F.fit_threshold_optimizer(1 - pv, 1 - yv, gv, "demographic_parity")
    approve_dp = pp.predict_approval(1 - p, g, seed=0)
    keep = ["A", "B"]
    gap_single = F.disparity_summary(1 - y, approve_single, g, keep)["dp_difference"]
    gap_dp = F.disparity_summary(1 - y, approve_dp, g, keep)["dp_difference"]
    assert gap_dp < 0.3 * gap_single
    assert set(pp.rules().group) == {"A", "B"}


def test_accuracy_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(4)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    t = F.accuracy_threshold(y, p)
    best = max(((p >= c) == y).mean() for c in np.append(p, 1.1))
    assert np.isclose(((p >= t) == y).mean(), best)


def test_metrics_holm_ece_wilson() -> None:
    assert np.allclose(M.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(5)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert M.ece(y, p) < 0.02 and M.ece(y, np.clip(p + 0.2, 0, 1)) > 0.15
    lo, hi = M.wilson(50, 100)
    assert lo < 0.5 < hi and np.isclose(lo + hi, 1.0)


def test_odds_ratio_recovers_planted_effect() -> None:
    rng = np.random.default_rng(6)
    n = 20000
    t = pd.DataFrame({"grp": rng.choice(["ref", "x"], n), "z": rng.normal(size=n)})
    logit = -2 + np.log(2.0) * (t.grp == "x") + 1.0 * t.z
    t["denied"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    r = odds_ratios(t, "grp", "ref", controls=t[["z"]]).iloc[0]
    assert r.ci_low < 2.0 < r.ci_high and r.converged
