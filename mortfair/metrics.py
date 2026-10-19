"""Classification metrics, calibration error, bootstrap confidence intervals and multiple-comparison correction."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the predicted probability."""
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile") -> float:
    """Expected calibration error: sum_b (n_b / n) |mean(y_b) - mean(p_b)| over quantile (or uniform) bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean predicted probability, observed rate and count per quantile bin (for calibration curves)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins = np.unique(idx)
    return (np.array([p[idx == b].mean() for b in bins]), np.array([y[idx == b].mean() for b in bins]),
            np.array([(idx == b).sum() for b in bins]))


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """ROC AUC, average precision, Brier score, log loss and ECE of probability scores."""
    return {"roc_auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)), "brier": brier(y, p),
            "log_loss": log_loss(y, p), "ece": ece(y, p)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k / n."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / d
    return (float(c - h), float(c + h))


def bootstrap_indices(n: int, n_boot: int, seed: int, strata: np.ndarray | None = None) -> np.ndarray:
    """Row indices of ``n_boot`` bootstrap resamples (optionally resampling within strata to keep their sizes)."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return rng.integers(0, n, size=(n_boot, n))
    out = np.empty((n_boot, n), dtype=np.int64)
    pos = 0
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        out[:, pos:pos + len(members)] = members[rng.integers(0, len(members), size=(n_boot, len(members)))]
        pos += len(members)
    return out


def paired_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], metric: Callable[[np.ndarray, np.ndarray], float],
                     reference: str, idx: np.ndarray) -> dict[str, dict[str, float]]:
    """Percentile CIs of a metric per model and of its difference to ``reference``, with a two-sided bootstrap p-value.

    The same resamples are used for every model, so the differences are paired.
    """
    vals = {m: np.array([metric(y[i], s[i]) for i in idx]) for m, s in scores.items()}
    out = {}
    for m, v in vals.items():
        d = v - vals[reference]
        p = 1.0 if m == reference else float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        out[m] = {"ci_low": float(np.quantile(v, 0.025)), "ci_high": float(np.quantile(v, 0.975)),
                  "diff_ci_low": float(np.quantile(d, 0.025)), "diff_ci_high": float(np.quantile(d, 0.975)),
                  "p_value": max(p, 1.0 / len(idx)) if m != reference else 1.0}
    return out


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (NaN when either input is constant)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)
