"""
src/modeling/baseline.py
────────────────────────
Quick baseline models for sanity-checking before full training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_baselines(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Train simple baselines and return metrics.

    Models
    ------
    1. Logistic Regression (class_weight='balanced')
    2. Random Forest (100 trees, class_weight='balanced')

    Returns
    -------
    dict[model_name → {"roc_auc": …, "pr_auc": …}]
    """
    results: dict[str, dict[str, float]] = {}

    # ── Logistic Regression ─────────────────────────────────
    lr = LogisticRegression(
        class_weight="balanced", max_iter=300, solver="lbfgs", n_jobs=-1
    )
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_valid)[:, 1]
    results["logistic_regression"] = {
        "roc_auc": float(roc_auc_score(y_valid, lr_prob)),
        "pr_auc": float(average_precision_score(y_valid, lr_prob)),
    }
    logger.info("LR  — ROC-AUC=%.4f  PR-AUC=%.4f", results["logistic_regression"]["roc_auc"], results["logistic_regression"]["pr_auc"])

    # ── Random Forest ───────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_valid)[:, 1]
    results["random_forest"] = {
        "roc_auc": float(roc_auc_score(y_valid, rf_prob)),
        "pr_auc": float(average_precision_score(y_valid, rf_prob)),
    }
    logger.info("RF  — ROC-AUC=%.4f  PR-AUC=%.4f", results["random_forest"]["roc_auc"], results["random_forest"]["pr_auc"])

    return results
