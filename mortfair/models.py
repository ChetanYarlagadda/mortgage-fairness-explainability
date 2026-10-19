"""Denial models: a debt-to-income lookup baseline, logistic regression, and LightGBM / XGBoost / CatBoost tuned with
Optuna on validation log loss (a proper scoring rule, so the tuned models stay calibrated)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from mortfair.features import BINARY, CATEGORICAL, FLAGS, NUMERIC
from mortfair.metrics import log_loss

THREADS = 2
LOG_FEATURES = ["loan_amount_k", "property_value_k", "income_k", "loan_to_income"]
# Denial risk jumps above 45-50% DTI and above 95% CLTV, so the logistic baseline gets cubic splines with knots there.
SPLINE_KNOTS = {"dti": [0.0, 20, 30, 36, 42, 46, 50, 60, 70], "cltv": [0.0, 60, 75, 80, 85, 90, 95, 100, 200]}
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Tuned:
    """Best model of one family, its parameters, validation log loss and the full trial history."""

    family: str
    model: Any
    params: dict
    val_log_loss: float
    best_iteration: int
    trials: pd.DataFrame
    seconds: float


# ------------------------------------------------------------------------------------------------------ baselines
class DtiLookup:
    """Baseline: denial rate of the training applications in the same debt-to-income bucket (missing is a bucket)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DtiLookup":
        key = self._key(X)
        df = pd.DataFrame({"k": key, "y": y})
        prior = float(np.mean(y))
        g = df.groupby("k").y.agg(["sum", "size"])
        self.rates_ = ((g["sum"] + prior * 5) / (g["size"] + 5)).to_dict()  # light smoothing toward the prior
        self.prior_ = prior
        return self

    @staticmethod
    def _key(X: pd.DataFrame) -> np.ndarray:
        return np.where(X["dti_missing"].to_numpy() > 0, -1.0, X["dti"].to_numpy())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Two-column probabilities like scikit-learn."""
        p = np.array([self.rates_.get(k, self.prior_) for k in self._key(X)])
        return np.column_stack([1 - p, p])


def _log1p_pos(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))


def logistic_pipeline(c: float, n_levels: dict[str, int]) -> Any:
    """Logistic regression: splines for DTI and CLTV, log-scaled amounts, standardised numerics, binary flags as is,
    one-hot categoricals."""
    cats = [list(range(n_levels[c_])) for c_ in CATEGORICAL]
    splines = [(f"spline_{f}", SplineTransformer(degree=3, knots=np.array(k).reshape(-1, 1), extrapolation="constant"),
                [f]) for f, k in SPLINE_KNOTS.items()]
    pre = ColumnTransformer([
        *splines,
        ("log", make_pipeline(FunctionTransformer(_log1p_pos), StandardScaler()), LOG_FEATURES),
        ("num", StandardScaler(), [c_ for c_ in NUMERIC if c_ not in LOG_FEATURES + list(SPLINE_KNOTS)]),
        ("bin", "passthrough", BINARY + FLAGS),
        ("cat", OneHotEncoder(categories=cats, handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(pre, LogisticRegression(C=c, max_iter=2000))


def fit_logistic(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, n_levels: dict[str, int],
                 cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)) -> Tuned:
    """Logistic regression with C chosen on validation log loss."""
    t0 = time.perf_counter()
    rows, best = [], (np.inf, None, None)
    for c in cs:
        pipe = logistic_pipeline(c, n_levels).fit(Xtr, ytr)
        ll = log_loss(yva, pipe.predict_proba(Xva)[:, 1])
        rows.append({"number": len(rows), "value": ll, "params_C": c})
        if ll < best[0]:
            best = (ll, pipe, c)
    return Tuned("Logistic regression", best[1], {"C": best[2]}, best[0], 0, pd.DataFrame(rows),
                 time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------- gradient boosting
def _study(objective: Callable[[optuna.Trial], float], n_trials: int, seed: int, timeout: float | None) -> optuna.Study:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=False)
    return study


def lgb_params(trial: optuna.Trial, seed: int) -> dict:
    """LightGBM search space."""
    return {"objective": "binary", "metric": "binary_logloss", "verbose": -1, "num_threads": THREADS, "seed": seed,
            "bagging_freq": 1, "max_cat_to_onehot": 8, "cat_smooth": 10.0,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True)}


def fit_lightgbm(params: dict, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame | None = None,
                 yva: np.ndarray | None = None, rounds: int = 2000, weight: np.ndarray | None = None) -> lgb.Booster:
    """Train LightGBM (categoricals by integer code); early stopping on (Xva, yva) when given."""
    cats = [c for c in CATEGORICAL if c in Xtr.columns]
    # feature_pre_filter=False: otherwise sparse binary flags (e.g. 'DTI missing') can be silently dropped when a
    # trial's min_child_samples exceeds their count
    dtr = lgb.Dataset(Xtr, ytr, weight=weight, categorical_feature=cats, free_raw_data=False,
                      params={"verbose": -1, "num_threads": THREADS, "feature_pre_filter": False})
    if Xva is None:
        return lgb.train(params, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cats, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def tune_lightgbm(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, n_trials: int,
                  seed: int = 0, timeout: float | None = None) -> Tuned:
    """Optuna (TPE) search over LightGBM; each trial early-stops on validation log loss."""
    t0 = time.perf_counter()
    best: dict[str, Any] = {"ll": np.inf}

    def objective(trial: optuna.Trial) -> float:
        params = lgb_params(trial, seed)
        booster = fit_lightgbm(params, Xtr, ytr, Xva, yva)
        ll = log_loss(yva, booster.predict(Xva, num_iteration=booster.best_iteration))
        trial.set_user_attr("best_iteration", booster.best_iteration)
        if ll < best["ll"]:
            best.update(ll=ll, model=booster, params=params, it=booster.best_iteration)
        return ll

    study = _study(objective, n_trials, seed, timeout)
    return Tuned("LightGBM", best["model"], best["params"], best["ll"], best["it"], study.trials_dataframe(),
                 time.perf_counter() - t0)


def xgb_frame(X: pd.DataFrame, n_levels: dict[str, int]) -> pd.DataFrame:
    """XGBoost reads categoricals from pandas category dtype (fixed category sets)."""
    Xc = X.copy()
    for c in CATEGORICAL:
        Xc[c] = pd.Categorical(Xc[c].astype(int), categories=range(n_levels[c]))
    return Xc


def tune_xgboost(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, n_levels: dict[str, int],
                 n_trials: int, seed: int = 0, timeout: float | None = None) -> Tuned:
    """Optuna search over XGBoost (hist, native categoricals); early stopping on validation log loss."""
    t0 = time.perf_counter()
    dtr = xgb.DMatrix(xgb_frame(Xtr, n_levels), ytr, enable_categorical=True, nthread=THREADS)
    dva = xgb.DMatrix(xgb_frame(Xva, n_levels), yva, enable_categorical=True, nthread=THREADS)
    best: dict[str, Any] = {"ll": np.inf}

    def objective(trial: optuna.Trial) -> float:
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "nthread": THREADS,
                  "seed": seed, "max_cat_to_onehot": 8,
                  "eta": trial.suggest_float("eta", 0.02, 0.2, log=True),
                  "max_depth": trial.suggest_int("max_depth", 3, 9),
                  "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                  "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True)}
        bst = xgb.train(params, dtr, num_boost_round=2000, evals=[(dva, "val")], early_stopping_rounds=50,
                        verbose_eval=False)
        ll = log_loss(yva, bst.predict(dva, iteration_range=(0, bst.best_iteration + 1)))
        trial.set_user_attr("best_iteration", bst.best_iteration)
        if ll < best["ll"]:
            best.update(ll=ll, model=bst, params=params, it=bst.best_iteration)
        return ll

    study = _study(objective, n_trials, seed, timeout)
    return Tuned("XGBoost", best["model"], best["params"], best["ll"], best["it"], study.trials_dataframe(),
                 time.perf_counter() - t0)


def cat_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categorical columns as integers."""
    Xc = X.copy()
    for c in CATEGORICAL:
        Xc[c] = Xc[c].astype(int)
    return Xc


def tune_catboost(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, n_trials: int,
                  seed: int = 0, timeout: float | None = None, max_rounds: int = 1500) -> Tuned:
    """Optuna search over CatBoost (ordered target statistics for categoricals); early stopping on validation log loss."""
    t0 = time.perf_counter()
    cat_idx = [list(Xtr.columns).index(c) for c in CATEGORICAL]
    ptr = Pool(cat_frame(Xtr), ytr, cat_features=cat_idx, thread_count=THREADS)
    pva = Pool(cat_frame(Xva), yva, cat_features=cat_idx, thread_count=THREADS)
    best: dict[str, Any] = {"ll": np.inf}

    def objective(trial: optuna.Trial) -> float:
        params = {"iterations": max_rounds, "loss_function": "Logloss", "od_type": "Iter", "od_wait": 50,
                  "thread_count": THREADS, "random_seed": seed, "verbose": False, "allow_writing_files": False,
                  "use_best_model": True,
                  "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                  "depth": trial.suggest_int("depth", 4, 8),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True)}
        model = CatBoostClassifier(**params)
        model.fit(ptr, eval_set=pva)
        ll = log_loss(yva, model.predict_proba(pva)[:, 1])
        trial.set_user_attr("best_iteration", model.get_best_iteration())
        if ll < best["ll"]:
            best.update(ll=ll, model=model, params=params, it=model.get_best_iteration())
        return ll

    study = _study(objective, n_trials, seed, timeout)
    return Tuned("CatBoost", best["model"], best["params"], best["ll"], best["it"], study.trials_dataframe(),
                 time.perf_counter() - t0)


def predict(tuned: Tuned, X: pd.DataFrame, n_levels: dict[str, int]) -> np.ndarray:
    """Predicted denial probability of a tuned model."""
    if tuned.family == "LightGBM":
        return tuned.model.predict(X, num_iteration=tuned.best_iteration)
    if tuned.family == "XGBoost":
        d = xgb.DMatrix(xgb_frame(X, n_levels), enable_categorical=True, nthread=THREADS)
        return tuned.model.predict(d, iteration_range=(0, tuned.best_iteration + 1))
    if tuned.family == "CatBoost":
        return tuned.model.predict_proba(cat_frame(X))[:, 1]
    return tuned.model.predict_proba(X)[:, 1]


def refit_params(tuned: Tuned) -> tuple[dict, int]:
    """LightGBM parameters and round count to refit the tuned configuration on other folds without early stopping."""
    return dict(tuned.params), max(1, int(tuned.best_iteration))


def tree_shap(booster: lgb.Booster, X: pd.DataFrame, num_iteration: int | None = None) -> tuple[np.ndarray, float]:
    """Exact path-dependent TreeSHAP (log-odds) from LightGBM's own implementation, and the expected value."""
    c = np.asarray(booster.predict(X, num_iteration=num_iteration, pred_contrib=True))
    return c[:, :-1], float(c[0, -1])
