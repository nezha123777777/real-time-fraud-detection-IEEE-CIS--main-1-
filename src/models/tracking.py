"""Optional MLflow tracking helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def log_training_run(
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[str | Path] | None = None,
) -> str | None:
    """Log a training run to MLflow if enabled in config."""
    cfg = load_config("model").get("mlflow", {})
    if not cfg.get("enabled", False):
        logger.info("MLflow disabled.")
        return None

    try:
        import mlflow
        import mlflow.xgboost
    except ImportError:
        logger.warning("MLflow is not installed; skipping tracking.")
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or cfg.get(
        "experiment_name",
        "ieee-fraud-detection",
    )
    mlflow.set_experiment(experiment_name)

    flattened_params = _flatten_dict(params)
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }

    run_name = os.getenv("MLFLOW_RUN_NAME") or cfg.get("run_name", "xgboost_smote_onnx")

    with mlflow.start_run(run_name=run_name) as run:
        if flattened_params:
            mlflow.log_params(flattened_params)
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        # MLflow 3.x: 'artifact_path' is deprecated → use 'name'.
        # Also set _estimator_type so XGBoost sklearn wrapper serialises correctly.
        register_name = cfg.get("register_model_name")
        try:
            if not hasattr(model, "_estimator_type"):
                model._estimator_type = "classifier"
            log_kwargs: dict[str, Any] = {"name": "model"}
            if register_name:
                log_kwargs["registered_model_name"] = register_name
            mlflow.xgboost.log_model(model, **log_kwargs)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "mlflow.xgboost.log_model failed (%s); "
                "falling back to logging pre-saved artifacts only.",
                exc,
            )

        for artifact in artifacts or []:
            path = Path(artifact)
            if path.exists():
                mlflow.log_artifact(str(path))

        logger.info("Logged MLflow run -> %s", run.info.run_id)
        return run.info.run_id


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        joined = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_dict(value, joined))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[joined] = value
    return result
