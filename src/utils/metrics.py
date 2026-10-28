"""
src/utils/metrics.py
────────────────────
Fraud-focused evaluation metrics.

All functions accept arrays of true labels and predicted probabilities
(or predicted labels where noted).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full metric suite for a binary fraud classifier.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth labels (0 / 1).
    y_prob : array-like of float
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold for converting probabilities to labels.

    Returns
    -------
    dict[str, float]
        Keys: ``roc_auc``, ``pr_auc``, ``precision``, ``recall``,
        ``f1``, ``threshold``.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }

    logger.info(
        "Metrics @ threshold=%.3f — ROC-AUC=%.4f  PR-AUC=%.4f  "
        "Precision=%.4f  Recall=%.4f  F1=%.4f",
        threshold,
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    min_t: float = 0.05,
    max_t: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    """Grid-search the decision threshold that maximises *metric*.

    Parameters
    ----------
    y_true, y_prob : array-like
    metric : str
        One of ``"f1"``, ``"precision"``, ``"recall"``.
    min_t, max_t, step : float
        Search grid boundaries and step size.

    Returns
    -------
    (best_threshold, best_score)
    """
    metric_fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
    if metric not in metric_fn:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(metric_fn)}")

    fn = metric_fn[metric]
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    best_t, best_s = 0.5, 0.0
    for t in np.arange(min_t, max_t + step, step):
        y_pred = (y_prob >= t).astype(int)
        s = fn(y_true, y_pred, zero_division=0)
        if s > best_s:
            best_s = s
            best_t = float(t)

    logger.info("Best threshold for %s: %.3f (score=%.4f)", metric, best_t, best_s)
    return best_t, best_s


def get_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return the 2×2 confusion matrix at the given threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true), y_pred)


def get_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) arrays."""
    return precision_recall_curve(np.asarray(y_true), np.asarray(y_prob))


def get_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) arrays."""
    return roc_curve(np.asarray(y_true), np.asarray(y_prob))
