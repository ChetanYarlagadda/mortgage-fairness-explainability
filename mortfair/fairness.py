"""Group fairness audit and mitigation.

Conventions: the model scores the probability of denial; fairness metrics are stated for the favourable outcome,
approval. Demographic parity compares approval rates across groups; equal opportunity compares the true approval
rate, i.e. the share of applicants the lender approved whom the model would also approve (1 - false denial rate).
Within-group calibration compares each group's observed denial rate with the mean predicted denial probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equal_opportunity_difference, selection_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from mortfair.metrics import ece, wilson


def _boot_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean of ``values`` (rows of a 2-D array are averaged column-wise)."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def group_table(y: np.ndarray, p: np.ndarray, deny: np.ndarray, groups: np.ndarray, reference: str,
                n_boot: int = 1000, seed: int = 0, min_auc_class: int = 10) -> pd.DataFrame:
    """Per-group audit of a denial model with bootstrap CIs for the gaps to the reference group.

    Columns: observed denial rate, mean predicted denial probability, observed minus predicted (calibration gap),
    AUC and ECE within the group, model approval rate (1 - share flagged for denial), true approval rate among
    applicants the lender approved, denial recall, and each rate's difference to ``reference``.
    """
    y, p, deny, groups = np.asarray(y), np.asarray(p, float), np.asarray(deny, bool), np.asarray(groups)
    rng = np.random.default_rng(seed)
    boots: dict[str, np.ndarray] = {}
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        yg, pg, dg = y[m], p[m], deny[m]
        n, k = int(m.sum()), int(yg.sum())
        lo, hi = wilson(k, n)
        appr = yg == 0
        # columns: observed denial, predicted denial prob, model approval, approved-and-model-approves (per approved)
        cols = np.column_stack([yg, pg, ~dg]).astype(float)
        b = _boot_means(cols, n_boot, rng)
        b_tar = _boot_means((~dg[appr]).astype(float), n_boot, rng) if appr.any() else np.full(n_boot, np.nan)
        boots[g] = np.column_stack([b, b_tar])
        gap = b[:, 0] - b[:, 1]
        p_gap = max(1.0 / n_boot, min(1.0, 2 * min((gap <= 0).mean(), (gap >= 0).mean())))
        rows.append({
            "group": g, "n": n, "n_denied": k, "observed_denial_rate": k / n, "observed_ci_low": lo,
            "observed_ci_high": hi, "mean_predicted": float(pg.mean()), "calibration_gap": float(yg.mean() - pg.mean()),
            "calibration_gap_ci_low": float(np.quantile(b[:, 0] - b[:, 1], 0.025)),
            "calibration_gap_ci_high": float(np.quantile(b[:, 0] - b[:, 1], 0.975)), "calibration_gap_p": p_gap,
            "observed_to_predicted": float(yg.mean() / pg.mean()),
            "roc_auc": float(roc_auc_score(yg, pg)) if min(k, n - k) >= min_auc_class else np.nan,
            "ece": ece(yg, pg, n_bins=5) if k >= min_auc_class else np.nan,
            "model_approval_rate": float((~dg).mean()), "true_approval_rate": float((~dg[appr]).mean()),
            "denial_recall": float(dg[~appr].mean()) if k else np.nan})
    out = pd.DataFrame(rows)
    ref = boots[reference]
    for name, j in [("observed_denial_rate", 0), ("mean_predicted", 1), ("model_approval_rate", 2),
                    ("true_approval_rate", 3)]:
        r0 = float(out.loc[out.group == reference, name].iloc[0])
        out[f"{name}_diff"] = out[name] - r0
        out[f"{name}_diff_ci_low"] = [float(np.quantile(boots[g][:, j] - ref[:, j], 0.025)) for g in out.group]
        out[f"{name}_diff_ci_high"] = [float(np.quantile(boots[g][:, j] - ref[:, j], 0.975)) for g in out.group]
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def disparity_summary(approved_true: np.ndarray, approve_pred: np.ndarray, groups: np.ndarray,
                      keep: list[str]) -> dict[str, float]:
    """Demographic parity and equal opportunity gaps (max - min over ``keep`` groups), computed with fairlearn."""
    m = np.isin(groups, keep)
    yt, yp, g = approved_true[m].astype(int), approve_pred[m].astype(int), groups[m]
    mf = MetricFrame(metrics={"approval_rate": selection_rate}, y_true=yt, y_pred=yp, sensitive_features=g)
    rates = mf.by_group["approval_rate"]
    return {"dp_difference": float(demographic_parity_difference(yt, yp, sensitive_features=g)),
            "dp_ratio": float(rates.min() / rates.max()),
            "eo_difference": float(equal_opportunity_difference(yt, yp, sensitive_features=g)),
            "lowest_approval_group": str(rates.idxmin()), "highest_approval_group": str(rates.idxmax())}


def reweighing_weights(groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kamiran-Calders reweighing: w(g, y) = P(g) P(y) / P(g, y), so group and label are independent when weighted."""
    df = pd.DataFrame({"g": groups, "y": y})
    pg = df.g.value_counts(normalize=True)
    py = df.y.value_counts(normalize=True)
    pgy = df.groupby(["g", "y"]).size() / len(df)
    return np.array([pg[g] * py[v] / pgy[(g, v)] for g, v in zip(df.g, df.y)])


class _ScoreColumn(ClassifierMixin, BaseEstimator):
    """Pre-fitted 'estimator' whose prediction is the score already stored in the first column of X."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_ScoreColumn":
        self.fitted_ = True
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X)[:, 0]


@dataclass
class PostProcessed:
    """Fitted group-specific (randomised) thresholds."""

    optimizer: ThresholdOptimizer
    constraint: str

    def predict_approval(self, approve_score: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
        """Approve (1) / deny (0) decisions for new applications."""
        X = pd.DataFrame({"score": approve_score})
        return np.asarray(self.optimizer.predict(X, sensitive_features=groups, random_state=seed)).astype(int)

    def rules(self) -> pd.DataFrame:
        """Per-group thresholds on the approval score (a mix of two thresholds when the rule is randomised)."""
        rows = []
        for g, d in self.optimizer.interpolated_thresholder_.interpolation_dict.items():
            rows.append({"group": g, "p0": float(d["p0"]), "threshold0": float(d["operation0"].threshold),
                         "p1": float(d["p1"]), "threshold1": float(d["operation1"].threshold)})
        return pd.DataFrame(rows)


def fit_threshold_optimizer(approve_score: np.ndarray, approved: np.ndarray, groups: np.ndarray,
                            constraint: str) -> PostProcessed:
    """fairlearn ThresholdOptimizer on validation scores (objective: accuracy). Uses the group at decision time."""
    X = pd.DataFrame({"score": approve_score})
    est = _ScoreColumn().fit(X, approved)
    to = ThresholdOptimizer(estimator=est, constraints=constraint, objective="accuracy_score", prefit=True,
                            predict_method="predict", grid_size=1000)
    to.fit(X, approved.astype(int), sensitive_features=groups)
    return PostProcessed(to, constraint)


def accuracy_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Single denial threshold that maximises accuracy (ties: the lowest such threshold)."""
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    # flag the top-k for every k: correct = denied flagged + approved not flagged
    tp = np.concatenate([[0], np.cumsum(ys)])
    fp = np.concatenate([[0], np.cumsum(1 - ys)])
    correct = tp + ((1 - ys).sum() - fp)
    k = int(np.argmax(correct))
    ps = np.asarray(p)[order]
    return float(ps[k - 1]) if k > 0 else float(ps[0] + 1e-9)
