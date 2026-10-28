"""Isolation Forest helpers for anomaly pre-filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.config import load_config, resolve_path
from src.utils.io import load_artifact, save_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_isolation_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> IsolationForest | None:
    """Train an Isolation Forest on legit training rows only."""
    if cfg is None:
        model_cfg = load_config("model")
        cfg = dict(model_cfg.get("isolation_forest", {}))

    if not cfg.get("enabled", False):
        logger.info("Isolation Forest disabled.")
        return None

    y = np.asarray(y_train)
    legit_mask = y == 0
    if legit_mask.sum() == 0:
        raise ValueError("Isolation Forest requires at least one legit training sample.")

    model = IsolationForest(
        contamination=cfg.get("contamination", 0.035),
        n_estimators=cfg.get("n_estimators", 200),
        max_samples=cfg.get("max_samples", "auto"),
        random_state=cfg.get("random_state", 42),
        n_jobs=cfg.get("n_jobs", -1),
    )
    model.fit(X_train.loc[legit_mask])
    logger.info("Trained Isolation Forest on %d legit rows.", int(legit_mask.sum()))
    return model


def score_anomalies(
    model: IsolationForest,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return anomaly scores and anomaly flags."""
    scores = model.decision_function(X)
    flags = model.predict(X) == -1
    return scores.astype(float), flags.astype(bool)


def save_isolation_forest(
    model: IsolationForest,
    name: str = "isolation_forest",
    directory: str | Path | None = None,
) -> Path:
    """Persist an Isolation Forest artefact."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["models"]["final_dir"])
    path = Path(directory) / f"{name}.joblib"
    return save_artifact(model, path)


def load_isolation_forest(
    name: str = "isolation_forest",
    directory: str | Path | None = None,
) -> IsolationForest:
    """Load a persisted Isolation Forest artefact."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["models"]["final_dir"])
    path = Path(directory) / f"{name}.joblib"
    return load_artifact(path)
