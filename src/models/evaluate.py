"""Evaluation helpers: compute metrics, pick thresholds, and optionally save PR/ROC/confusion plots and feature importance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.metrics import (
    compute_all_metrics,
    find_best_threshold,
    get_confusion_matrix,
    get_pr_curve,
    get_roc_curve,
)

logger = get_logger(__name__)


def evaluate_model(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    split_name: str = "valid",
    threshold: float | None = None,
    save_plots: bool = True,
) -> dict[str, float]:
    """Run the full evaluation suite on a single split.

    Parameters
    ----------
    model : trained XGBClassifier
    X : feature matrix
    y : true labels
    split_name : str
        Used in filenames (``"valid"`` or ``"test"``).
    threshold : float, optional
        Decision threshold.  If ``None``, the optimal threshold is found
        automatically (maximising F1 by default).
    save_plots : bool
        Whether to save PR / ROC curves and confusion matrix images.

    Returns
    -------
    dict[str, float]  — full metric dictionary (also saved as JSON).
    """
    model_cfg = load_config("model")
    paths_cfg = load_config("paths")
    fig_dir = resolve_path(paths_cfg["reports"]["figures_dir"])
    met_dir = resolve_path(paths_cfg["reports"]["metrics_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(y)
    y_prob = model.predict_proba(X)[:, 1]

    # ── Find optimal threshold if not provided ──────────────
    if threshold is None:
        thresh_cfg = model_cfg.get("thresholding", {})
        threshold, _ = find_best_threshold(
            y, y_prob,
            metric=thresh_cfg.get("optimise_metric", "f1"),
            min_t=thresh_cfg.get("min_threshold", 0.05),
            max_t=thresh_cfg.get("max_threshold", 0.95),
            step=thresh_cfg.get("step", 0.01),
        )

    # ── Compute metrics ─────────────────────────────────────
    metrics = compute_all_metrics(y, y_prob, threshold=threshold)
    metrics["split"] = split_name

    # ── Save metrics as JSON ────────────────────────────────
    metrics_path = met_dir / f"metrics_{split_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics → %s", metrics_path)

    # ── Save plots ──────────────────────────────────────────
    if save_plots:
        _save_pr_curve(y, y_prob, fig_dir, split_name)
        _save_roc_curve(y, y_prob, fig_dir, split_name)
        _save_confusion_matrix(y, y_prob, threshold, fig_dir, split_name)
        _save_feature_importance(model, X.columns.tolist(), fig_dir, split_name)

    return metrics


# ── Plot helpers ────────────────────────────────────────────────


def _save_pr_curve(y, y_prob, fig_dir: Path, split: str) -> None:
    precision, recall, _ = get_pr_curve(y, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({split})")
    ax.grid(alpha=0.3)
    path = fig_dir / f"fig_pr_curve_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curve → %s", path)


def _save_roc_curve(y, y_prob, fig_dir: Path, split: str) -> None:
    fpr, tpr, _ = get_roc_curve(y, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({split})")
    ax.grid(alpha=0.3)
    path = fig_dir / f"fig_roc_curve_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve → %s", path)


def _save_confusion_matrix(y, y_prob, threshold, fig_dir: Path, split: str) -> None:
    cm = get_confusion_matrix(y, y_prob, threshold)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({split}, t={threshold:.3f})")
    fig.colorbar(im, ax=ax)
    path = fig_dir / f"fig_confusion_matrix_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", path)


def _save_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    fig_dir: Path,
    split: str,
    top_n: int = 30,
) -> None:
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    ax.barh(
        [feature_names[i] for i in indices],
        importance[indices],
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Features ({split})")
    path = fig_dir / f"fig_feature_importance_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance → %s", path)
