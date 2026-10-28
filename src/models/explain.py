"""SHAP explainability helpers for fraud models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_tree_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for an XGBoost classifier."""
    return shap.TreeExplainer(model)


def explain_prediction(
    model: xgb.XGBClassifier,
    X_single: pd.DataFrame,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Explain a single prediction with SHAP."""
    if len(X_single) != 1:
        raise ValueError("explain_prediction expects exactly one row.")

    model_cfg = load_config("model")
    top_n = top_n or int(model_cfg.get("explainability", {}).get("top_n", 10))

    explainer = build_tree_explainer(model)
    explanation = explainer(X_single)

    shap_values = np.asarray(explanation.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    shap_row = shap_values[0]

    base_values = np.asarray(explanation.base_values)
    if base_values.ndim > 1:
        base_value = float(base_values[0, -1])
    else:
        base_value = float(np.ravel(base_values)[0])

    feature_values = X_single.iloc[0].to_dict()
    contributions = []
    for feature_name, shap_value in zip(X_single.columns, shap_row):
        raw_value = feature_values.get(feature_name)
        if isinstance(raw_value, (np.generic,)):
            raw_value = raw_value.item()
        if pd.isna(raw_value):
            raw_value = None
        contributions.append({
            "feature": feature_name,
            "feature_value": raw_value,
            "shap_value": float(shap_value),
            "abs_shap_value": float(abs(shap_value)),
        })

    contributions.sort(key=lambda item: item["abs_shap_value"], reverse=True)
    return {
        "base_value": base_value,
        "top_features": contributions[:top_n],
        "feature_count": len(contributions),
    }


def save_global_shap_importance(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    split_name: str = "valid",
    top_n: int | None = None,
    max_samples: int = 2000,
) -> Path:
    """Save a global SHAP importance CSV and bar chart for a feature matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_cfg = load_config("model")
    explain_cfg = model_cfg.get("explainability", {})
    top_n = top_n or int(explain_cfg.get("top_n", 10))

    paths_cfg = load_config("paths")
    fig_dir = resolve_path(paths_cfg["reports"]["figures_dir"])
    met_dir = resolve_path(paths_cfg["reports"]["metrics_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    if len(X) > max_samples:
        X_eval = X.sample(n=max_samples, random_state=42)
    else:
        X_eval = X

    explainer = build_tree_explainer(model)
    explanation = explainer(X_eval)
    shap_values = np.asarray(explanation.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    importance = pd.DataFrame({
        "feature": X_eval.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = met_dir / f"shap_global_importance_{split_name}.csv"
    importance.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    plot_df = importance.head(top_n).sort_values("mean_abs_shap", ascending=True)
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global SHAP Importance ({split_name})")
    fig.tight_layout()
    fig_path = fig_dir / f"fig_shap_global_importance_{split_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved SHAP global importance -> %s", csv_path)
    logger.info("Saved SHAP global importance plot -> %s", fig_path)
    return csv_path
