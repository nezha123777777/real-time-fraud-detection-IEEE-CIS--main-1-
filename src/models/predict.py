"""Prediction helpers for trained models (probabilities and labels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_proba(
    model: xgb.XGBClassifier, X: pd.DataFrame
) -> np.ndarray:
    """Return fraud probabilities (column 1 of predict_proba)."""
    probs = model.predict_proba(X)[:, 1]
    logger.info("Generated probabilities for %d samples.", len(probs))
    return probs


def predict_labels(
    model: xgb.XGBClassifier, X: pd.DataFrame, threshold: float = 0.5
) -> np.ndarray:
    """Return binary labels at the given threshold."""
    probs = predict_proba(model, X)
    labels = (probs >= threshold).astype(int)
    n_fraud = int(labels.sum())
    logger.info(
        "Predicted %d fraud out of %d (%.2f%%) at threshold=%.3f",
        n_fraud, len(labels), 100 * n_fraud / len(labels), threshold,
    )
    return labels
