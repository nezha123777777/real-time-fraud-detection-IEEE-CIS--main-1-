"""ONNX export and inference helpers for XGBoost models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_expected_columns(path: str | Path) -> list[str] | None:
    """Load the feature schema saved during training."""
    path = Path(path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").splitlines()


def align_expected_columns(
    X: pd.DataFrame,
    expected_cols: Sequence[str] | None,
) -> pd.DataFrame:
    """Add missing columns as NaN and reorder to match the training schema."""
    if not expected_cols:
        return X

    X_aligned = X.copy()
    for col in expected_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = np.nan
    return X_aligned[list(expected_cols)]


def save_model_onnx(
    model: xgb.XGBClassifier,
    feature_names: Sequence[str] | None = None,
    name: str = "xgb_final",
    directory: str | Path | None = None,
) -> Path:
    """Export a trained XGBoost classifier to ONNX."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["models"]["final_dir"])

    n_features = len(feature_names) if feature_names is not None else getattr(
        model, "n_features_in_", None
    )
    if n_features is None:
        raise ValueError("Could not determine the number of input features for ONNX export.")

    booster = model.get_booster()
    original_feature_names = booster.feature_names
    original_feature_types = booster.feature_types
    booster.feature_names = None
    booster.feature_types = None
    try:
        onnx_model = convert_xgboost(
            model,
            initial_types=[("input", FloatTensorType([None, n_features]))],
        )
    finally:
        booster.feature_names = original_feature_names
        booster.feature_types = original_feature_types

    path = Path(directory) / f"{name}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_model.SerializeToString())
    logger.info("Saved ONNX model -> %s", path)
    return path


def load_onnx_session(
    model_path: str | Path,
    providers: list[str] | None = None,
) -> ort.InferenceSession:
    """Load an ONNX Runtime session from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {path}")

    session = ort.InferenceSession(
        str(path),
        providers=providers or ["CPUExecutionProvider"],
    )
    logger.info("Loaded ONNX model <- %s", path)
    return session


def predict_proba_onnx(
    session: ort.InferenceSession,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Return positive-class probabilities from an ONNX classifier."""
    X_input = _to_onnx_input(X)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(None, {input_name: X_input})

    prob_idx = output_names.index("probabilities") if "probabilities" in output_names else -1
    prob_output = np.asarray(outputs[prob_idx])

    if prob_output.ndim == 1:
        probs = prob_output
    elif prob_output.ndim == 2 and prob_output.shape[1] >= 2:
        probs = prob_output[:, 1]
    else:
        raise ValueError(
            f"Unexpected ONNX probability output shape: {prob_output.shape}"
        )

    logger.info("Generated ONNX probabilities for %d samples.", len(probs))
    return probs.astype(np.float64, copy=False)


def _to_onnx_input(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Cast feature input to float32 for ONNX Runtime."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)
