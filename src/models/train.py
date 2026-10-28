"""Training helpers for IEEE-CIS fraud detection.

Includes feature-matrix prep, optional SMOTE resampling, and XGBoost fit/save helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Compute ``n_negatives / n_positives`` for imbalance handling."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0:
        raise ValueError("No positive samples in training target.")
    spw = n_neg / n_pos
    logger.info("scale_pos_weight = %.2f  (neg=%d, pos=%d)", spw, n_neg, n_pos)
    return spw


def _prepare_xgboost_params(
    y_train: pd.Series | np.ndarray,
    params_override: dict[str, Any] | None = None,
    xgb_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Build XGBoost params for a specific training target distribution."""
    if xgb_cfg is None:
        model_cfg = load_config("model")
        xgb_cfg = dict(model_cfg.get("xgboost", {}))
    else:
        xgb_cfg = dict(xgb_cfg)

    early_stopping = xgb_cfg.pop("early_stopping_rounds", 50)

    if params_override:
        xgb_cfg.update(params_override)

    if xgb_cfg.get("scale_pos_weight") is None:
        xgb_cfg["scale_pos_weight"] = _compute_scale_pos_weight(y_train)

    return xgb_cfg, early_stopping


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    smote_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Optionally rebalance the training split with SMOTE.

    SMOTE is applied only to the training data after preprocessing, never
    to validation or test data.
    """
    y = np.asarray(y_train)

    if smote_cfg is None:
        model_cfg = load_config("model")
        smote_cfg = dict(model_cfg.get("smote", {}))

    if not smote_cfg.get("enabled", False):
        logger.info("SMOTE disabled - using original training distribution.")
        return X_train, y

    class_counts = pd.Series(y).value_counts().sort_index()
    if class_counts.shape[0] < 2:
        raise ValueError("SMOTE requires at least two classes in y_train.")
    if class_counts.nunique() == 1:
        logger.info(
            "SMOTE enabled but classes are already balanced: %s - skipping resampling.",
            class_counts.to_dict(),
        )
        return X_train, y

    minority_count = int(class_counts.min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least 2 samples in the minority class.")

    requested_k = int(smote_cfg.get("k_neighbors", 5))
    effective_k = min(requested_k, minority_count - 1)
    if effective_k < 1:
        raise ValueError("SMOTE could not determine a valid k_neighbors value.")
    if effective_k != requested_k:
        logger.info(
            "Reducing SMOTE k_neighbors from %d to %d because the minority class has %d samples.",
            requested_k,
            effective_k,
            minority_count,
        )

    logger.info("Applying SMOTE on training data only - before: %s", class_counts.to_dict())
    from imblearn.over_sampling import SMOTE

    sampler = SMOTE(
        sampling_strategy=smote_cfg.get("sampling_strategy", "auto"),
        random_state=smote_cfg.get("random_state", 42),
        k_neighbors=effective_k,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train, y)

    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)

    y_resampled = np.asarray(y_resampled)
    logger.info(
        "SMOTE complete - after: %s",
        pd.Series(y_resampled).value_counts().sort_index().to_dict(),
    )
    return X_resampled, y_resampled


def prepare_feature_matrix(
    df: pd.DataFrame, drop_cols: list[str] | None = None
) -> pd.DataFrame:
    """Drop non-feature columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    drop_cols : list[str], optional
        Columns to exclude.  Defaults to ``configs/model.yaml →
        drop_before_training``.

    Returns
    -------
    pd.DataFrame  — feature-only matrix.
    """
    if drop_cols is None:
        model_cfg = load_config("model")
        drop_cols = model_cfg.get("drop_before_training", [])

    existing = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing)

    # Drop remaining object / category columns that weren't encoded
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        logger.warning(
            "Dropping %d un-encoded object/category columns: %s", len(obj_cols), obj_cols[:10]
        )
        X = X.drop(columns=obj_cols)

    return X


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | np.ndarray | None = None,
    params_override: dict[str, Any] | None = None,
    fit_verbose: bool | int = 100,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with early stopping.

    Parameters
    ----------
    X_train, y_train : training features and labels.
    X_valid, y_valid : validation features and labels for early stopping.
        If not provided, early stopping is disabled.
    params_override : dict, optional
        Override any XGBoost parameter from ``configs/model.yaml``.

    Returns
    -------
    xgb.XGBClassifier  — the trained model.
    """
    xgb_params, early_stopping = _prepare_xgboost_params(
        y_train,
        params_override=params_override,
    )

    if X_valid is not None and y_valid is not None and early_stopping is not None:
        xgb_params["early_stopping_rounds"] = early_stopping

    params_for_log = {
        k: v for k, v in xgb_params.items() if k != "scale_pos_weight"
    }
    logger.info("Training XGBoost with params: %s", params_for_log)

    model = xgb.XGBClassifier(**xgb_params)

    fit_kwargs: dict[str, Any] = {}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["verbose"] = fit_verbose

    model.fit(
        X_train,
        y_train,
        **fit_kwargs,
    )

    logger.info("Training complete — best iteration: %s", getattr(model, "best_iteration", "N/A"))
    return model


def save_model(
    model: xgb.XGBClassifier,
    name: str = "xgb_final",
    directory: str | Path | None = None,
) -> Path:
    """Save a trained model to XGBoost's native JSON format.

    Parameters
    ----------
    model : trained XGBClassifier
    name : str
        Filename stem (e.g. ``"xgb_v01"``).
    directory : path, optional
        Defaults to ``artifacts/models/final/``.

    Returns
    -------
    Path  — saved file path.
    """
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["models"]["final_dir"])
    path = Path(directory) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(path)
    logger.info("Saved XGBoost model -> %s", path)
    return path


def load_model(
    name: str = "xgb_final",
    directory: str | Path | None = None,
) -> xgb.XGBClassifier:
    """Load a trained model from XGBoost's native JSON format."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["models"]["final_dir"])
    path = Path(directory) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = xgb.XGBClassifier()
    model.load_model(path)
    logger.info("Loaded XGBoost model <- %s", path)
    return model
