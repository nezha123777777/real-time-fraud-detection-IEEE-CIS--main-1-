"""Threshold optimisation utilities with load/save helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.metrics import find_best_threshold

logger = get_logger(__name__)


def optimise_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Find and return optimal threshold(s) from model config.

    Returns
    -------
    dict with keys ``"threshold"``, ``"metric"``, ``"score"``.
    """
    model_cfg = load_config("model")
    thresh_cfg = model_cfg.get("thresholding", {})

    metric = thresh_cfg.get("optimise_metric", "f1")
    best_t, best_s = find_best_threshold(
        y_true, y_prob,
        metric=metric,
        min_t=thresh_cfg.get("min_threshold", 0.05),
        max_t=thresh_cfg.get("max_threshold", 0.95),
        step=thresh_cfg.get("step", 0.01),
    )
    return {"threshold": best_t, "metric": metric, "score": best_s}


def save_threshold(
    threshold_info: dict[str, float],
    directory: str | Path | None = None,
) -> Path:
    """Persist threshold to JSON."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["artifacts"]["thresholds_dir"])
    path = Path(directory) / "threshold.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(threshold_info, fh, indent=2)
    logger.info("Saved threshold → %s", path)
    return path


def load_threshold(directory: str | Path | None = None) -> dict[str, float]:
    """Load previously saved threshold."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["artifacts"]["thresholds_dir"])
    path = Path(directory) / "threshold.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
