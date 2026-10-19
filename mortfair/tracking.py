"""MLflow experiment tracking (local SQLite store under ./mlruns, git-ignored)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

EXPERIMENT = "mortgage-fairness-explainability"


def init_mlflow(root: Path) -> None:
    """Point MLflow at <root>/mlruns unless MLFLOW_TRACKING_URI is already set."""
    store = root / "mlruns"
    store.mkdir(parents=True, exist_ok=True)
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(f"sqlite:///{(store / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def log_result(run_name: str, params: dict, metrics: dict[str, float]) -> None:
    """Log one model configuration as its own MLflow run."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v)[:500] for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
