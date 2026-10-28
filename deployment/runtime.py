"""Shared inference runtime helpers for FastAPI and streaming consumers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.aggregation_features import load_agg_maps  # noqa: E402
from src.features.build_features import (  # noqa: E402
    build_features_stateless,
    transform_stateful_features,
)
from src.models.anomaly import load_isolation_forest, score_anomalies  # noqa: E402
from src.models.explain import explain_prediction  # noqa: E402
from src.models.onnx_utils import (  # noqa: E402
    align_expected_columns,
    load_expected_columns,
    load_onnx_session,
    predict_proba_onnx,
)
from src.models.thresholding import load_threshold  # noqa: E402
from src.models.train import load_model, prepare_feature_matrix  # noqa: E402
from src.preprocessing.categorical_encoding import load_encoders  # noqa: E402
from src.preprocessing.missing_values import load_imputers, transform_impute  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def load_artifacts(project_root: Path | None = None) -> dict[str, Any]:
    """Load all inference artefacts into memory."""
    paths_cfg = load_config("paths")
    model_cfg = load_config("model")
    final_dir = resolve_path(paths_cfg["models"]["final_dir"])

    onnx_model_path = final_dir / "xgb_final.onnx"
    onnx_model = load_onnx_session(onnx_model_path)
    native_model = load_model(directory=final_dir)

    threshold, review_threshold = _load_thresholds()
    expected_columns = load_expected_columns(
        resolve_path(paths_cfg["artifacts"]["feature_lists_dir"]) / "features.txt"
    )

    isolation_model = None
    if model_cfg.get("isolation_forest", {}).get("enabled", False):
        isolation_path = final_dir / "isolation_forest.joblib"
        if isolation_path.exists():
            isolation_model = load_isolation_forest(directory=final_dir)

    return {
        "model": onnx_model,
        "native_model": native_model,
        "threshold": threshold,
        "review_threshold": review_threshold,
        "imputers": load_imputers(),
        "encoders": load_encoders(),
        "agg_maps": load_agg_maps(),
        "expected_columns": expected_columns,
        "model_version": onnx_model_path.stem,
        "isolation_model": isolation_model,
    }


def preprocess_payload(payload: dict[str, Any], artifacts: dict[str, Any]) -> pd.DataFrame:
    """Convert raw JSON payload into a model-ready feature matrix."""
    df_raw = pd.DataFrame([payload])
    df = build_features_stateless(df_raw)
    df = transform_impute(df, artifacts["imputers"])

    freq_enc = artifacts["encoders"].get("frequency_encoder") if artifacts.get("encoders") else None
    if freq_enc:
        df = freq_enc.transform(df)

    df = transform_stateful_features(df, {"agg_maps": artifacts.get("agg_maps", {})})
    X = prepare_feature_matrix(df, drop_cols=["TransactionDT"])
    return align_expected_columns(X, artifacts.get("expected_columns"))


def predict_payload(payload: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    """Run end-to-end inference for a single transaction payload."""
    X = preprocess_payload(payload, artifacts)
    return _predict_from_features(X, artifacts)


def explain_payload(payload: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    """Run prediction plus SHAP explanation for a single payload."""
    X = preprocess_payload(payload, artifacts)
    prediction = _predict_from_features(X, artifacts)
    explanation = explain_prediction(artifacts["native_model"], X)
    return {
        **prediction,
        **explanation,
    }


def _predict_from_features(X: pd.DataFrame, artifacts: dict[str, Any]) -> dict[str, Any]:
    """Score an already-preprocessed feature matrix."""
    proba = float(predict_proba_onnx(artifacts["model"], X)[0])
    threshold = float(artifacts.get("threshold", 0.5))
    review_threshold = float(artifacts.get("review_threshold", 0.35))

    anomaly_score = None
    anomaly_flag = False
    isolation_model = artifacts.get("isolation_model")
    if isolation_model is not None:
        scores, flags = score_anomalies(isolation_model, X)
        anomaly_score = float(scores[0])
        anomaly_flag = bool(flags[0])

    action = _resolve_action(
        proba=proba,
        threshold=threshold,
        review_threshold=review_threshold,
        anomaly_flag=anomaly_flag,
    )

    return {
        "fraud_probability": proba,
        "fraud_label": int(proba >= threshold),
        "action": action,
        "threshold": threshold,
        "review_threshold": review_threshold,
        "anomaly_flag": anomaly_flag,
        "anomaly_score": anomaly_score,
        "model_version": artifacts.get("model_version", "unknown"),
    }


def _load_thresholds() -> tuple[float, float]:
    """Resolve block and review thresholds from env, artefacts, and config."""
    model_cfg = load_config("model")
    saved_threshold = 0.5
    try:
        thresh_info = load_threshold()
        saved_threshold = float(thresh_info.get("threshold", 0.5))
    except Exception:
        pass

    threshold = float(os.getenv("FRAUD_THRESHOLD", saved_threshold))
    default_review = float(model_cfg.get("thresholding", {}).get("review_threshold", 0.35))
    review_threshold = float(os.getenv("FRAUD_REVIEW_THRESHOLD", default_review))
    review_threshold = min(review_threshold, threshold)
    return threshold, review_threshold


def _resolve_action(
    proba: float,
    threshold: float,
    review_threshold: float,
    anomaly_flag: bool,
) -> str:
    """Map model outputs to a three-level action."""
    if proba >= threshold:
        return "BLOCK"
    if proba >= review_threshold or anomaly_flag:
        return "REVIEW"
    return "ALLOW"
