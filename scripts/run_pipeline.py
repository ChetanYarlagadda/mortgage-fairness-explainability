"""Run every stage end to end.

1. Data: download 2025 Arizona home-purchase records from the FFIEC HMDA Data Browser API, check counts and loan-amount
   sums against the API's own aggregations, filter to conventional, first-lien, owner-occupied, site-built, single-unit
   applications that were originated or denied, and build the typed application table.
2. Models (rotation 0 of a 5-fold train / validation / test rotation): DTI lookup and logistic baselines, LightGBM /
   XGBoost / CatBoost tuned with Optuna on validation log loss, compared on the untouched test fold.
3. Cross-fitting: the tuned LightGBM configuration is refit in all five rotations, so every application gets a denial
   probability, a decision, TreeSHAP values and mitigated decisions from models that never saw it.
4. Fairness audit by race, ethnicity, sex and age; mitigation by reweighing and fairlearn ThresholdOptimizer.
5. Observational context: unadjusted and adjusted denial odds ratios; how well the model's inputs predict the
   protected attributes (proxy check); feature-group ablation.
6. Explanations: TreeSHAP (checked against the shap library), LIME on a few cases, SHAP-LIME agreement, and SHAP's top
   factor against the denial reasons lenders reported.
Tables -> reports/results, figures -> reports/figures, runs -> MLflow (./mlruns).

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --quick    # smoke test on a 25% subsample with tiny budgets (data/cache/smoke/)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from mortfair import data as D  # noqa: E402
from mortfair import disparity as DS  # noqa: E402
from mortfair import explain as EX  # noqa: E402
from mortfair import fairness as F  # noqa: E402
from mortfair import metrics as M  # noqa: E402
from mortfair import models as MD  # noqa: E402
from mortfair.features import (ATTRIBUTES, BINARY, CATEGORICAL, EXCLUDED, FEATURES, FLAGS, GROUP_OF, GROUPS,  # noqa: E402
                               REFERENCE, Encoder, application_table, missingness_by_outcome)

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
REF = ROOT / "data" / "reference"
VERSION = "v2"  # bump to invalidate every cached model output
FAMILIES = ("LightGBM", "XGBoost", "CatBoost")
BASE, RW, TO_DP, TO_EO = ("Baseline (unaware)", "Reweighing", "Group thresholds: demographic parity",
                          "Group thresholds: equal opportunity")


@dataclass(frozen=True)
class Settings:
    """Sampling, tuning, evaluation and explanation settings of one run."""

    seed: int = 0
    n_folds: int = 5
    subsample: float = 1.0
    top_lenders: int = 40
    trials: tuple[tuple[str, int], ...] = (("LightGBM", 30), ("XGBoost", 20), ("CatBoost", 10))
    timeout_s: tuple[tuple[str, float], ...] = (("LightGBM", 150.0), ("XGBoost", 150.0), ("CatBoost", 200.0))
    lr_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    n_boot: int = 1000
    n_boot_mitigation: int = 500
    min_group_n: int = 300
    mitigation_attribute: str = "race"
    shap_rows: int = 3000
    shap_check_rows: int = 200
    lime_per_stratum: int = 10
    lime_samples: int = 5000
    lime_seeds: tuple[int, ...] = (0, 1)
    proxy_rounds: int = 300


QUICK = Settings(subsample=0.25, trials=(("LightGBM", 2), ("XGBoost", 2), ("CatBoost", 2)), n_boot=50,
                 n_boot_mitigation=30, min_group_n=100, shap_rows=300, shap_check_rows=50, lime_per_stratum=2,
                 lime_samples=500, proxy_rounds=50)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Timer:
    """Wall-clock seconds per stage."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def run(self, name: str, fn: Callable):
        t0 = time.perf_counter()
        out = fn()
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0
        log(f"{name}: {self.stages[name]:.1f}s")
        return out


def key_of(spec: dict) -> str:
    """Short hash of everything that changes an output."""
    return hashlib.sha1(json.dumps({**spec, "version": VERSION}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cached(path: Path, fn: Callable):
    """Load a pickled result if present, else compute and store it."""
    if path.exists():
        log(f"cache hit {path.name}")
        return pickle.loads(path.read_bytes())
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def write(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / name, index=False, float_format="%.6g")


# ------------------------------------------------------------------------------------------------------------ data
def stage_data(S: Settings, out: Path) -> tuple[pd.DataFrame, dict]:
    """Download, cross-check, filter and type the applications; write the data tables."""
    dl = D.download(RAW)
    raw = D.load_raw(dl.path)
    agg = D.fetch_aggregations(REF / "api_aggregations_AZ_2025_purpose1.json")
    check = D.crosscheck(raw, agg)
    write(check, out, "api_crosscheck.csv")
    pop, steps = D.filter_population(raw)
    write(steps, out, "population_filter.csv")
    t = application_table(pop)
    if S.subsample < 1:
        t = t.sample(frac=S.subsample, random_state=S.seed).reset_index(drop=True)
    write(missingness_by_outcome(t), out, "missingness_by_outcome.csv")
    write(pd.DataFrame([{"field": k, "reason": v} for k, v in EXCLUDED.items()]), out, "excluded_fields.csv")
    den = t[t.denied == 1]
    first = den.denial_reason_1.value_counts()
    anyr = pd.Series([r for rs in den.denial_reasons for r in rs.split("|") if r]).value_counts()
    write(pd.DataFrame({"reason": first.index, "first_reason": first.values,
                        "any_reason": anyr.reindex(first.index).fillna(0).astype(int).values,
                        "share_first": first.values / len(den)}), out, "denial_reasons.csv")
    prov = {"url": dl.url, "final_url": dl.final_url, "sha256": dl.sha256, "sha256_expected": D.EXPECTED_SHA256,
            "sha256_matches": dl.sha256 == D.EXPECTED_SHA256, "bytes": dl.n_bytes, "last_modified": dl.last_modified,
            "raw_rows": len(raw), "population_rows": len(pop), "analysed_rows": len(t),
            "denied": int(t.denied.sum()), "denial_rate": float(t.denied.mean()),
            "lenders": int(t.lender_id.nunique()), "api_checks": len(check),
            "api_count_matches": int(check.count_match.sum()), "api_max_sum_rel_diff": float(check.sum_rel_diff.max())}
    log(f"data: {len(raw):,} raw rows -> {len(t):,} applications, {prov['denied']:,} denied "
        f"({prov['denial_rate']:.2%}); API check {prov['api_count_matches']}/{prov['api_checks']} counts match")
    return t, prov


def make_folds(t: pd.DataFrame, S: Settings) -> np.ndarray:
    """Fold id per application, stratified by outcome and race group."""
    key = t.denied.astype(str) + "|" + t.race
    folds = np.zeros(len(t), dtype=int)
    skf = StratifiedKFold(n_splits=S.n_folds, shuffle=True, random_state=S.seed)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(t)), key)):
        folds[te] = k
    return folds


def rotation(folds: np.ndarray, r: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation r: test = fold r, validation = fold r+1 (mod k), training = the other folds."""
    te = np.flatnonzero(folds == r)
    va = np.flatnonzero(folds == (r + 1) % k)
    tr = np.flatnonzero((folds != r) & (folds != (r + 1) % k))
    return tr, va, te


def encode(t: pd.DataFrame, tr: np.ndarray, S: Settings) -> tuple[Encoder, pd.DataFrame, dict[str, int]]:
    enc = Encoder(top_lenders=S.top_lenders).fit(t.iloc[tr])
    X = enc.transform(t)
    return enc, X, {c: enc.n_levels(c) for c in CATEGORICAL}


# ---------------------------------------------------------------------------------------------------------- models
def stage_models(t: pd.DataFrame, folds: np.ndarray, S: Settings, key: str) -> dict:
    """Tune and fit every model family on rotation 0; return validation and test predictions."""
    tr, va, te = rotation(folds, 0, S.n_folds)
    _, X, n_levels = encode(t, tr, S)
    y = t.denied.to_numpy()
    trials, timeouts = dict(S.trials), dict(S.timeout_s)

    def run() -> dict:
        tuned = {}
        lookup = MD.DtiLookup().fit(X.iloc[tr], y[tr])
        tuned["Logistic regression"] = MD.fit_logistic(X.iloc[tr], y[tr], X.iloc[va], y[va], n_levels, S.lr_cs)
        log(f"logistic: C={tuned['Logistic regression'].params['C']}")
        tuned["LightGBM"] = MD.tune_lightgbm(X.iloc[tr], y[tr], X.iloc[va], y[va], trials["LightGBM"], S.seed,
                                             timeouts["LightGBM"])
        log(f"LightGBM: {len(tuned['LightGBM'].trials)} trials, val log loss {tuned['LightGBM'].val_log_loss:.4f}")
        tuned["XGBoost"] = MD.tune_xgboost(X.iloc[tr], y[tr], X.iloc[va], y[va], n_levels, trials["XGBoost"], S.seed,
                                           timeouts["XGBoost"])
        log(f"XGBoost: {len(tuned['XGBoost'].trials)} trials, val log loss {tuned['XGBoost'].val_log_loss:.4f}")
        tuned["CatBoost"] = MD.tune_catboost(X.iloc[tr], y[tr], X.iloc[va], y[va], trials["CatBoost"], S.seed,
                                             timeouts["CatBoost"])
        log(f"CatBoost: {len(tuned['CatBoost'].trials)} trials, val log loss {tuned['CatBoost'].val_log_loss:.4f}")
        preds = {"DTI lookup": (lookup.predict_proba(X.iloc[va])[:, 1], lookup.predict_proba(X.iloc[te])[:, 1])}
        for name, tu in tuned.items():
            preds[name] = (MD.predict(tu, X.iloc[va], n_levels), MD.predict(tu, X.iloc[te], n_levels))
        summary = {name: {k: v for k, v in tu.__dict__.items() if k != "model"} for name, tu in tuned.items()}
        return {"preds": preds, "tuned": summary, "lgb_model": tuned["LightGBM"].model.model_to_string()}

    return cached(CACHE / f"models-{key}.pkl", run)


def model_tables(res: dict, t: pd.DataFrame, folds: np.ndarray, S: Settings, out: Path) -> pd.DataFrame:
    """Test-fold comparison with paired bootstrap CIs and Holm-corrected tests against logistic regression."""
    _, va, te = rotation(folds, 0, S.n_folds)
    y_va, y_te = t.denied.to_numpy()[va], t.denied.to_numpy()[te]
    order = ["DTI lookup", "Logistic regression", "LightGBM", "XGBoost", "CatBoost"]
    scores = {m: res["preds"][m][1] for m in order}
    idx = M.bootstrap_indices(len(y_te), S.n_boot, S.seed)
    rows, sig = [], []
    boot = {name: M.paired_bootstrap(y_te, scores, fn, "Logistic regression", idx)
            for name, fn in [("roc_auc", roc_auc_score), ("ap", average_precision_score), ("log_loss", M.log_loss)]}
    for m in order:
        r = {"model": m, "val_log_loss": M.log_loss(y_va, res["preds"][m][0]), **M.score_metrics(y_te, scores[m])}
        for name, b in boot.items():
            r[f"{name}_ci_low"], r[f"{name}_ci_high"] = b[m]["ci_low"], b[m]["ci_high"]
            if m != "Logistic regression":
                sig.append({"model": m, "metric": name, "diff": r[name] - M.score_metrics(y_te, scores[
                    "Logistic regression"])[name], "diff_ci_low": b[m]["diff_ci_low"],
                    "diff_ci_high": b[m]["diff_ci_high"], "p_value": b[m]["p_value"]})
        thr = F.accuracy_threshold(y_va, res["preds"][m][0])
        flag = scores[m] >= thr
        r.update({"threshold": thr, "flagged_share": float(flag.mean()), "accuracy": float((flag == y_te).mean()),
                  "denial_recall": float(flag[y_te == 1].mean()),
                  "denial_precision": float(y_te[flag].mean()) if flag.any() else np.nan, "test_rows": len(y_te),
                  "test_denied": int(y_te.sum())})
        rows.append(r)
    for name, fn in [("roc_auc", roc_auc_score), ("ap", average_precision_score), ("log_loss", M.log_loss)]:
        b = M.paired_bootstrap(y_te, {m: scores[m] for m in FAMILIES}, fn, "LightGBM", idx)
        for m in ("XGBoost", "CatBoost"):
            sig.append({"model": m, "reference": "LightGBM", "metric": name, "diff": fn(y_te, scores[m]) - fn(y_te, scores[
                "LightGBM"]), "diff_ci_low": b[m]["diff_ci_low"], "diff_ci_high": b[m]["diff_ci_high"],
                "p_value": b[m]["p_value"]})
    table = pd.DataFrame(rows)
    sig = pd.DataFrame(sig)
    sig["reference"] = sig["reference"].fillna("Logistic regression")
    sig["p_holm"] = M.holm(sig.p_value.tolist())
    write(table, out, "model_results.csv")
    write(sig, out, "model_significance.csv")
    cal = []
    for m in order:
        mp, obs, n = M.reliability_bins(y_te, scores[m], 10)
        cal += [{"model": m, "bin": i, "mean_predicted": a, "observed": b, "n": int(c)}
                for i, (a, b, c) in enumerate(zip(mp, obs, n))]
    write(pd.DataFrame(cal), out, "calibration_curves.csv")
    tuned = []
    trials = []
    for name, tu in res["tuned"].items():
        tuned.append({"model": name, "trials": len(tu["trials"]), "val_log_loss": tu["val_log_loss"],
                      "best_iteration": tu["best_iteration"], "tuning_seconds": tu["seconds"],
                      "params": json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in tu["params"].items()
                                            if k not in ("verbose", "allow_writing_files")})})
        tr_df = tu["trials"].copy()
        tr_df.insert(0, "model", name)
        trials.append(tr_df[[c for c in tr_df.columns if c in ("model", "number", "value") or c.startswith("params_")
                             or c == "user_attrs_best_iteration"]])
    write(pd.DataFrame(tuned), out, "tuned_models.csv")
    write(pd.concat(trials, ignore_index=True), out, "tuning_trials.csv")
    return table


# --------------------------------------------------------------------------------------------------- cross-fitting
def stage_crossfit(t: pd.DataFrame, folds: np.ndarray, S: Settings, params: dict, key: str) -> dict:
    """Refit the tuned LightGBM in every rotation; out-of-fold scores, decisions, SHAP and mitigated decisions."""

    def run() -> dict:
        n, y, race = len(t), t.denied.to_numpy(), t[S.mitigation_attribute].to_numpy()
        o = {k: np.full(n, np.nan) for k in ["p", "thr", "p_rw", "thr_rw", "approve_dp", "approve_eo"]}
        o["shap"] = np.zeros((n, len(FEATURES)), dtype=np.float32)
        o["shap_base"] = np.zeros(n)
        rules, iters = [], []
        for r in range(S.n_folds):
            tr, va, te = rotation(folds, r, S.n_folds)
            _, X, _ = encode(t, tr, S)
            b = MD.fit_lightgbm(params, X.iloc[tr], y[tr], X.iloc[va], y[va])
            p_va, p_te = b.predict(X.iloc[va], num_iteration=b.best_iteration), b.predict(X.iloc[te], num_iteration=b.best_iteration)
            o["p"][te], o["thr"][te] = p_te, F.accuracy_threshold(y[va], p_va)
            sv, base = MD.tree_shap(b, X.iloc[te], b.best_iteration)
            o["shap"][te], o["shap_base"][te] = sv, base
            w = F.reweighing_weights(race[tr], y[tr])
            brw = MD.fit_lightgbm(params, X.iloc[tr], y[tr], X.iloc[va], y[va], weight=w)
            prw_va = brw.predict(X.iloc[va], num_iteration=brw.best_iteration)
            o["p_rw"][te] = brw.predict(X.iloc[te], num_iteration=brw.best_iteration)
            o["thr_rw"][te] = F.accuracy_threshold(y[va], prw_va)
            for name, constraint in [("approve_dp", "demographic_parity"), ("approve_eo", "true_positive_rate_parity")]:
                pp = F.fit_threshold_optimizer(1 - p_va, 1 - y[va], race[va], constraint)
                o[name][te] = pp.predict_approval(1 - p_te, race[te], seed=S.seed + r)
                rr = pp.rules()
                rr.insert(0, "method", TO_DP if name == "approve_dp" else TO_EO)
                rr.insert(0, "rotation", r)
                rules.append(rr)
            iters.append({"rotation": r, "best_iteration": b.best_iteration,
                          "best_iteration_reweighed": brw.best_iteration, "threshold": o["thr"][te][0],
                          "threshold_reweighed": o["thr_rw"][te][0], "test_auc": roc_auc_score(y[te], p_te)})
            log(f"rotation {r}: AUC {iters[-1]['test_auc']:.4f}, threshold {iters[-1]['threshold']:.3f}")
        o["rules"] = pd.concat(rules, ignore_index=True)
        o["rotations"] = pd.DataFrame(iters)
        return o

    return cached(CACHE / f"crossfit-{key}.pkl", run)


# ---------------------------------------------------------------------------------------------- fairness & mitigation
def stage_audit(t: pd.DataFrame, cf: dict, S: Settings, out: Path) -> pd.DataFrame:
    """Per-group audit of the cross-fitted baseline for each protected attribute."""
    y, p = t.denied.to_numpy(), cf["p"]
    deny = p >= cf["thr"]
    tabs = []
    for a in ATTRIBUTES:
        g = t[a].to_numpy()
        tab = F.group_table(y, p, deny, g, REFERENCE[a], n_boot=S.n_boot, seed=S.seed)
        tab.insert(0, "attribute", a)
        tabs.append(tab)
        # the fairlearn MetricFrame must reproduce the approval rates computed by group_table
        keep = tab.group[tab.n >= S.min_group_n].tolist()
        summ = F.disparity_summary(1 - y, (~deny).astype(int), g, keep)
        rates = tab.set_index("group").loc[keep, "model_approval_rate"]
        assert abs(summ["dp_difference"] - (rates.max() - rates.min())) < 1e-9
    audit = pd.concat(tabs, ignore_index=True)
    # Holm correction of the calibration-gap tests over every group with at least 150 applications
    fam = audit.n >= 150
    audit["calibration_gap_p_holm"] = np.nan
    audit.loc[fam, "calibration_gap_p_holm"] = M.holm(audit.loc[fam, "calibration_gap_p"].tolist())
    write(audit, out, "fairness_audit.csv")
    cal = []
    for a in ATTRIBUTES:
        for g, d in t.assign(p=p).groupby(a):
            if len(d) < S.min_group_n:
                continue
            mp, obs, n = M.reliability_bins(d.denied.to_numpy(), d.p.to_numpy(), 5)
            cal += [{"attribute": a, "group": g, "bin": i, "mean_predicted": x, "observed": z, "n": int(c)}
                    for i, (x, z, c) in enumerate(zip(mp, obs, n))]
    write(pd.DataFrame(cal), out, "group_calibration.csv")
    return audit


def _rate_boot(approve_true: np.ndarray, approve_pred: np.ndarray, codes: np.ndarray,
               idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap approval rates and true approval rates per group code (arrays of shape n_boot x n_groups)."""
    G = codes.max() + 1
    appr, tar = np.empty((len(idx), G)), np.empty((len(idx), G))
    for b, i in enumerate(idx):
        c, a, at = codes[i], approve_pred[i], approve_true[i]
        appr[b] = np.bincount(c, weights=a, minlength=G) / np.maximum(np.bincount(c, minlength=G), 1)
        tar[b] = np.bincount(c, weights=at * a, minlength=G) / np.maximum(np.bincount(c, weights=at, minlength=G), 1)
    return appr, tar


def stage_mitigation(t: pd.DataFrame, cf: dict, S: Settings, out: Path) -> pd.DataFrame:
    """Accuracy and disparity of the baseline, reweighing and the two group-threshold rules (out-of-fold)."""
    y = t.denied.to_numpy()
    approved = 1 - y
    methods = {BASE: ((cf["p"] < cf["thr"]).astype(int), cf["p"]),
               RW: ((cf["p_rw"] < cf["thr_rw"]).astype(int), cf["p_rw"]),
               TO_DP: (cf["approve_dp"].astype(int), None), TO_EO: (cf["approve_eo"].astype(int), None)}
    idx = M.bootstrap_indices(len(y), S.n_boot_mitigation, S.seed + 7)
    rows, by_group = [], []
    for name, (appr, score) in methods.items():
        deny = 1 - appr
        r = {"method": name, "accuracy": float((appr == approved).mean()),
             "balanced_accuracy": float(0.5 * (deny[y == 1].mean() + appr[y == 0].mean())),
             "denial_recall": float(deny[y == 1].mean()), "denial_precision": float(y[deny == 1].mean()),
             "approval_rate": float(appr.mean()),
             "roc_auc": float(roc_auc_score(y, score)) if score is not None else np.nan,
             "uses_group_at_decision": name in (TO_DP, TO_EO), "uses_group_in_training": name == RW}
        acc_b = np.array([(appr[i] == approved[i]).mean() for i in idx])
        r["accuracy_ci_low"], r["accuracy_ci_high"] = np.quantile(acc_b, [0.025, 0.975])
        for a in ATTRIBUTES:
            g = t[a].to_numpy()
            sizes = pd.Series(g).value_counts()
            keep = sizes[sizes >= S.min_group_n].index.tolist()
            s = F.disparity_summary(approved, appr, g, keep)
            r[f"{a}_dp_difference"], r[f"{a}_dp_ratio"], r[f"{a}_eo_difference"] = (
                s["dp_difference"], s["dp_ratio"], s["eo_difference"])
            if a == S.mitigation_attribute:
                # CI of the gap between the two groups that define it in the point estimate (a resampled
                # max-minus-min would be biased upwards)
                cats = pd.Categorical(g)
                codes = cats.codes.astype(int)
                b_appr, b_tar = _rate_boot(approved.astype(float), appr.astype(float), codes, idx)
                for nm, b_rate, true_only in [("dp", b_appr, False), ("eo", b_tar, True)]:
                    mask = (y == 0) if true_only else np.ones(len(y), bool)
                    point = pd.Series(appr[mask]).groupby(g[mask]).mean().loc[keep]
                    hi_g, lo_g = point.idxmax(), point.idxmin()
                    d = b_rate[:, list(cats.categories).index(hi_g)] - b_rate[:, list(cats.categories).index(lo_g)]
                    r[f"{nm}_ci_low"], r[f"{nm}_ci_high"] = np.quantile(d, [0.025, 0.975])
                    r[f"{nm}_gap_groups"] = f"{hi_g} - {lo_g}"
            for grp in sizes.index:
                m = g == grp
                by_group.append({"method": name, "attribute": a, "group": grp, "n": int(m.sum()),
                                 "approval_rate": float(appr[m].mean()),
                                 "true_approval_rate": float(appr[m & (y == 0)].mean()),
                                 "denial_recall": float((1 - appr)[m & (y == 1)].mean()) if (m & (y == 1)).any() else np.nan})
        rows.append(r)
    res = pd.DataFrame(rows)
    write(res, out, "mitigation_results.csv")
    write(pd.DataFrame(by_group), out, "mitigation_by_group.csv")
    write(cf["rules"], out, "threshold_rules.csv")
    write(cf["rotations"], out, "crossfit_rotations.csv")
    return res


# --------------------------------------------------------------------------------------------- observational context
def stage_odds(t: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Unadjusted and adjusted denial odds ratios for every protected attribute."""
    specs = {"Unadjusted": None, "Application controls": DS.control_frame(t, lenders=False),
             "Application controls + lender": DS.control_frame(t, lenders=True)}
    rows = []
    for a in ATTRIBUTES:
        for spec, ctrl in specs.items():
            r = DS.odds_ratios(t, a, REFERENCE[a], ctrl)
            r.insert(0, "specification", spec)
            r.insert(0, "attribute", a)
            rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    write(res, out, "adjusted_odds_ratios.csv")
    return res


def stage_proxy(t: pd.DataFrame, folds: np.ndarray, S: Settings, out: Path) -> pd.DataFrame:
    """How well the model's inputs (no protected attributes) predict each protected attribute (test-fold AUC)."""
    tr, _, te = rotation(folds, 0, S.n_folds)
    _, X, _ = encode(t, tr, S)
    targets = {"Black (vs White)": (t.race.isin(["Black", "White"]), t.race == "Black"),
               "Hispanic or Latino (vs not)": (t.ethnicity.isin(["Hispanic or Latino", "Not Hispanic or Latino"]),
                                               t.ethnicity == "Hispanic or Latino"),
               "Female (vs male)": (t.sex.isin(["Female", "Male"]), t.sex == "Female"),
               "Age 62 or older": (t.age != "Not available", t.age_62_plus == 1)}
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50,
              "verbose": -1, "num_threads": MD.THREADS, "seed": S.seed}
    rows = []
    for name, (rows_ok, target) in targets.items():
        trm, tem = np.intersect1d(tr, np.flatnonzero(rows_ok)), np.intersect1d(te, np.flatnonzero(rows_ok))
        yt = target.to_numpy().astype(int)
        b = MD.fit_lightgbm(params, X.iloc[trm], yt[trm], rounds=S.proxy_rounds)
        p = b.predict(X.iloc[tem])
        mp = t.tract_minority_pct.to_numpy()[tem]
        okm = ~np.isnan(mp)
        rows.append({"target": name, "test_rows": len(tem), "positive_share": float(yt[tem].mean()),
                     "auc_model_features": float(roc_auc_score(yt[tem], p)),
                     "auc_tract_minority_share_only": float(roc_auc_score(yt[tem][okm], mp[okm]))})
        a_ = rows[-1]["auc_tract_minority_share_only"]
        rows[-1]["auc_tract_minority_share_either_direction"] = max(a_, 1 - a_)
    res = pd.DataFrame(rows)
    write(res, out, "proxy_auc.csv")
    return res


def stage_ablation(t: pd.DataFrame, folds: np.ndarray, S: Settings, params: dict, out: Path) -> pd.DataFrame:
    """Test-fold AUC / AP / log loss of the tuned LightGBM without each feature group (and with DTI alone)."""
    tr, va, te = rotation(folds, 0, S.n_folds)
    _, X, _ = encode(t, tr, S)
    y = t.denied.to_numpy()
    sets = {"All features": FEATURES, **{f"Without {g[0].lower() + g[1:]}": [f for f in FEATURES if f not in fs]
                                         for g, fs in GROUPS.items()},
            "Debt-to-income only": ["dti", "dti_missing"]}
    rows = []
    for name, cols in sets.items():
        b = MD.fit_lightgbm(params, X.iloc[tr][cols], y[tr], X.iloc[va][cols], y[va])
        p = b.predict(X.iloc[te][cols], num_iteration=b.best_iteration)
        rows.append({"feature_set": name, "n_features": len(cols), **M.score_metrics(y[te], p)})
    res = pd.DataFrame(rows)
    for c in ["roc_auc", "ap", "log_loss"]:
        res[f"{c}_change"] = res[c] - res.loc[0, c]
    write(res, out, "ablation.csv")
    return res


# ------------------------------------------------------------------------------------------------------ explanations
def stage_explain(t: pd.DataFrame, folds: np.ndarray, cf: dict, S: Settings, params: dict, key: str,
                  out: Path) -> dict:
    """SHAP check and global importance, reason-code match, and LIME vs SHAP agreement on selected cases."""
    import shap

    tr, va, te = rotation(folds, 0, S.n_folds)
    enc, X, _ = encode(t, tr, S)
    y = t.denied.to_numpy()
    b = MD.fit_lightgbm(params, X.iloc[tr], y[tr], X.iloc[va], y[va])
    it = b.best_iteration
    rng = np.random.default_rng(S.seed)
    # 1. TreeSHAP from LightGBM against the shap library on a few rows
    rows = X.iloc[te[: S.shap_check_rows]]
    own, _ = MD.tree_shap(b, rows, it)
    try:
        lib = np.asarray(shap.TreeExplainer(b).shap_values(rows))
        lib = lib[1] if lib.ndim == 3 and lib.shape[0] == 2 else lib
        check = {"rows": len(rows), "max_abs_diff": float(np.abs(lib - own).max()), "error": ""}
    except Exception as e:  # the comparison is reported, not required
        check = {"rows": len(rows), "max_abs_diff": np.nan, "error": f"{type(e).__name__}: {e}"[:300]}
    raw = b.predict(rows, num_iteration=it, raw_score=True)
    check["additivity_max_abs_err"] = float(np.abs(own.sum(1) + MD.tree_shap(b, rows, it)[1] - raw).max())
    write(pd.DataFrame([check]), out, "shap_check.csv")
    # 2. global importance on a sample of the test fold (OOF SHAP of rotation 0)
    samp = np.sort(rng.choice(te, size=min(S.shap_rows, len(te)), replace=False))
    sv = cf["shap"][samp]
    imp = pd.DataFrame({"feature": FEATURES, "group": [GROUP_OF[f] for f in FEATURES],
                        "mean_abs_shap": np.abs(sv).mean(0),
                        "mean_shap_denied": sv[y[samp] == 1].mean(0), "mean_shap_approved": sv[y[samp] == 0].mean(0)})
    write(imp.sort_values("mean_abs_shap", ascending=False), out, "shap_importance.csv")
    grp = EX.group_attributions(sv, FEATURES)
    write(pd.DataFrame({"group": grp.columns, "mean_abs_shap": grp.abs().mean().values}).sort_values(
        "mean_abs_shap", ascending=False), out, "shap_group_importance.csv")
    np.savez_compressed(out / "shap_sample.npz", shap=sv.astype(np.float32), X=X.iloc[samp].to_numpy(np.float32),
                        y=y[samp].astype(np.int8), features=np.array(FEATURES))
    # 3. reported denial reasons vs SHAP's top factor, all out-of-fold denials
    den = np.flatnonzero(y == 1)
    top = EX.top_denial_group(cf["shap"][den], FEATURES)
    reason = t.denial_reason_1.iloc[den].reset_index(drop=True)
    write(EX.reason_match(top, reason), out, "reason_match.csv")
    ct = pd.crosstab(reason, top)
    ct = ct.reindex(columns=list(GROUPS), fill_value=0)
    ct.insert(0, "n_denials", ct.sum(1))
    write(ct.reset_index().rename(columns={"denial_reason_1": "reported_reason"}), out, "reason_crosstab.csv")
    # 4. LIME on a few test cases vs SHAP, and LIME against itself with another seed
    thr = cf["thr"][te][0]
    p_te = cf["p"][te]
    y_te = y[te]
    k = S.lime_per_stratum
    near = te[np.argsort(np.abs(p_te - thr))[:k]]
    hi = np.setdiff1d(te[(y_te == 1) & (p_te >= thr)], near)
    low = np.setdiff1d(te[(y_te == 0) & (p_te < thr)], near)
    cases = np.concatenate([rng.choice(hi, k, replace=False), near, rng.choice(low, k, replace=False)])
    strata = ["Denied, flagged"] * k + ["Near the threshold"] * k + ["Approved, not flagged"] * k
    cat_idx = [FEATURES.index(f) for f in BINARY + FLAGS + CATEGORICAL]

    def predict_proba(A: np.ndarray) -> np.ndarray:
        p = b.predict(pd.DataFrame(A, columns=FEATURES).astype(np.float32), num_iteration=it)
        return np.column_stack([1 - p, p])

    def run_lime() -> dict:
        train = X.iloc[tr].to_numpy(np.float64)
        return {s: EX.lime_attributions(predict_proba, train, X.iloc[cases].to_numpy(np.float64), FEATURES, cat_idx,
                                        seed=s, num_samples=S.lime_samples) for s in S.lime_seeds}

    lime = cached(CACHE / f"lime-{key}.pkl", run_lime)
    shap_cases, _ = MD.tree_shap(b, X.iloc[cases], it)
    l0, l1 = lime[S.lime_seeds[0]], lime[S.lime_seeds[1]]
    agree = []
    no_flags = np.array([f not in FLAGS for f in FEATURES])
    for comp, a_, b_ in [("SHAP vs LIME", shap_cases, l0), ("LIME seed 0 vs seed 1", l0, l1),
                         ("SHAP vs LIME, missing-value flags excluded", shap_cases[:, no_flags], l0[:, no_flags])]:
        tab = EX.agreement_table(a_, b_)
        tab.insert(0, "comparison", comp)
        tab["stratum"] = [strata[i] for i in tab.case]
        agree.append(tab)
    agree = pd.concat(agree, ignore_index=True)
    write(agree, out, "explainer_agreement.csv")
    summ = agree.groupby(["comparison", "k"])[["top_k_overlap", "sign_agreement", "rank_corr_abs",
                                               "same_top_feature"]].mean().reset_index()
    write(summ, out, "explainer_agreement_summary.csv")
    write(agree.groupby(["comparison", "stratum", "k"])[["top_k_overlap", "rank_corr_abs"]].mean().reset_index(),
          out, "explainer_agreement_by_stratum.csv")

    def top3(a: np.ndarray) -> str:
        return "; ".join(f"{FEATURES[j]} ({'+' if a[j] > 0 else '-'})" for j in np.argsort(-np.abs(a))[:3])

    lc = pd.DataFrame({"case": range(len(cases)), "stratum": strata, "denied": y[cases], "p_denial": cf["p"][cases],
                       "reported_reason": t.denial_reason_1.iloc[cases].fillna("").to_numpy(),
                       "shap_top3": [top3(a) for a in shap_cases], "lime_top3": [top3(a) for a in l0],
                       "lime_seed1_top3": [top3(a) for a in l1]})
    write(lc, out, "lime_cases.csv")
    tops = [{"method": name, "stratum": strata[i], "top_feature": FEATURES[int(np.argmax(np.abs(a[i])))]}
            for name, a in [("SHAP", shap_cases), ("LIME", l0), ("LIME seed 1", l1)] for i in range(len(cases))]
    tops = pd.DataFrame(tops).groupby(["method", "top_feature"]).size().rename("cases").reset_index()
    write(tops.sort_values(["method", "cases"], ascending=[True, False]), out, "explainer_top_features.csv")
    np.savez_compressed(out / "local_attributions.npz", shap=shap_cases.astype(np.float32), lime=l0.astype(np.float32),
                        lime_seed1=l1.astype(np.float32), X=X.iloc[cases].to_numpy(np.float32),
                        p=cf["p"][cases].astype(np.float32), y=y[cases].astype(np.int8), features=np.array(FEATURES),
                        lender_levels=np.array(enc.level_names("lender")))
    return {"shap_check": check, "lime_cases": len(cases)}


# ------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test on a subsample with tiny budgets")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    S = QUICK if args.quick else Settings()
    out = (CACHE / "smoke" / "results") if args.quick else ROOT / "reports" / "results"
    fig_dir = (CACHE / "smoke" / "figures") if args.quick else ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    t0 = time.perf_counter()

    t, prov = timer.run("data", lambda: stage_data(S, out))
    folds = make_folds(t, S)
    data_key = {"sha256": prov["sha256"], "rows": len(t), "subsample": S.subsample, "seed": S.seed,
                "n_folds": S.n_folds, "top_lenders": S.top_lenders}
    mkey = key_of({**data_key, "trials": S.trials, "timeout": S.timeout_s, "cs": S.lr_cs})
    res = timer.run("models", lambda: stage_models(t, folds, S, mkey))
    table = model_tables(res, t, folds, S, out)
    log("\n" + table[["model", "val_log_loss", "roc_auc", "ap", "brier", "log_loss", "ece"]].to_string(index=False))
    params = res["tuned"]["LightGBM"]["params"]
    ckey = key_of({**data_key, "models": mkey, "attr": S.mitigation_attribute})
    cf = timer.run("crossfit", lambda: stage_crossfit(t, folds, S, params, ckey))
    audit = timer.run("audit", lambda: stage_audit(t, cf, S, out))
    mit = timer.run("mitigation", lambda: stage_mitigation(t, cf, S, out))
    odds = timer.run("odds_ratios", lambda: stage_odds(t, out))
    proxy = timer.run("proxy", lambda: stage_proxy(t, folds, S, out))
    abl = timer.run("ablation", lambda: stage_ablation(t, folds, S, params, out))
    ekey = key_of({**data_key, "models": mkey, "lime": [S.lime_per_stratum, S.lime_samples, S.lime_seeds]})
    exp = timer.run("explain", lambda: stage_explain(t, folds, cf, S, params, ekey, out))

    try:
        from mortfair import tracking
        tracking.init_mlflow(ROOT)
        for _, r in table.iterrows():
            tracking.log_result(f"model/{r.model}", {"model": r.model, **({"params": res["tuned"][r.model]["params"]}
                                if r.model in res["tuned"] else {})},
                                {k: r[k] for k in ["val_log_loss", "roc_auc", "ap", "brier", "log_loss", "ece"]})
        for _, r in mit.iterrows():
            tracking.log_result(f"mitigation/{r.method}", {"method": r.method, "attribute": S.mitigation_attribute},
                                {k: r[k] for k in ["accuracy", "balanced_accuracy", "approval_rate", "race_dp_difference",
                                                   "race_eo_difference", "roc_auc"]})
        mlflow_ok = True
    except Exception as e:  # tracking is optional for the results
        log(f"MLflow logging skipped: {e}")
        mlflow_ok = False

    import fairlearn
    import lightgbm
    import lime
    import optuna
    import shap
    import sklearn
    import xgboost
    import catboost
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "quick": args.quick,
            "settings": asdict(S), "data": prov, "cache_keys": {"models": mkey, "crossfit": ckey, "lime": ekey},
            "stage_seconds": {k: round(v, 1) for k, v in timer.stages.items()},
            "total_seconds": round(time.perf_counter() - t0, 1), "shap_check": exp["shap_check"],
            "mlflow_logged": mlflow_ok,
            "rows": {"audit": len(audit), "odds": len(odds), "proxy": len(proxy), "ablation": len(abl)},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                         "xgboost": xgboost.__version__, "catboost": catboost.__version__, "optuna": optuna.__version__,
                         "shap": shap.__version__, "lime": getattr(lime, "__version__", "0.2.0.1"),
                         "fairlearn": fairlearn.__version__}}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    if not args.no_figures:
        from mortfair.figures import make_all
        for p in make_all(out, fig_dir):
            log(f"wrote {p.name}")
    log(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
