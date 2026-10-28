"""
src/features/aggregation_features.py
─────────────────────────────────────
Group-level aggregation features (mean, std, count) for columns like
``card1``, ``addr1``, ``ProductCD``.

LEAKAGE PREVENTION
──────────────────
Aggregations such as mean ``TransactionAmt`` per card are computed on
the **training set only** and then mapped onto validation / test via a
lookup dictionary.  This prevents future-aware statistics from leaking
into the model.

Usage
-----
1. Call :func:`fit_aggregations` on the training DataFrame.
2. Save the resulting maps with :func:`save_agg_maps`.
3. Call :func:`transform_aggregations` on each split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import resolve_path, load_config
from src.utils.io import save_artifact, load_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default aggregation specs:  (group_col, agg_col, agg_func)
DEFAULT_AGG_SPECS: list[tuple[str, str, str]] = [
    ("card1", "TransactionAmt", "mean"),
    ("card1", "TransactionAmt", "std"),
    ("card1", "TransactionAmt", "count"),
    ("addr1", "TransactionAmt", "mean"),
    ("addr1", "TransactionAmt", "std"),
    ("ProductCD", "TransactionAmt", "mean"),
    ("ProductCD", "TransactionAmt", "count"),
]


def _agg_col_name(group: str, agg_col: str, func: str) -> str:
    """Canonical name for an aggregation feature."""
    return f"{group}_{agg_col}_{func}"


def fit_aggregations(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str]] | None = None,
) -> dict[str, dict]:
    """Compute aggregation lookup dictionaries from **training data**.

    Parameters
    ----------
    df : pd.DataFrame
        **Training** DataFrame.
    specs : list of (group_col, agg_col, agg_func), optional
        Override the default aggregation specifications.

    Returns
    -------
    dict  — keyed by output column name, value is ``{group_val: agg_value}``.
    """
    specs = specs or DEFAULT_AGG_SPECS
    agg_maps: dict[str, dict] = {}

    for group_col, agg_col, func in specs:
        if group_col not in df.columns or agg_col not in df.columns:
            logger.warning("Skipping agg (%s, %s, %s) — column missing.", group_col, agg_col, func)
            continue

        name = _agg_col_name(group_col, agg_col, func)
        mapping = df.groupby(group_col)[agg_col].agg(func).to_dict()
        agg_maps[name] = {"group_col": group_col, "mapping": mapping}

    logger.info("Fitted %d aggregation features.", len(agg_maps))
    return agg_maps


def transform_aggregations(
    df: pd.DataFrame,
    agg_maps: dict[str, dict],
) -> pd.DataFrame:
    """Apply pre-computed aggregation maps to any split.

    Unseen group values are filled with the global median of the mapping.
    """
    df = df.copy()

    for feat_name, info in agg_maps.items():
        group_col = info["group_col"]
        mapping = info["mapping"]

        if group_col not in df.columns:
            continue

        # Filter out NaN values from the mapping (e.g. std of single-element groups)
        valid_values = [v for v in mapping.values() if not (isinstance(v, float) and np.isnan(v))]
        fill_value = float(np.nanmedian(valid_values)) if valid_values else 0.0

        mapped = df[group_col].map(mapping)
        # Replace NaN from missing groups AND NaN from std-of-1 with fill_value
        df[feat_name] = mapped.fillna(fill_value).astype("float32")

    logger.info("Applied %d aggregation features.", len(agg_maps))
    return df


def save_agg_maps(agg_maps: dict, directory: str | Path | None = None) -> Path:
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["artifacts"]["feature_lists_dir"])
    return save_artifact(agg_maps, Path(directory) / "agg_maps.joblib")


def load_agg_maps(directory: str | Path | None = None) -> dict:
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["artifacts"]["feature_lists_dir"])
    return load_artifact(Path(directory) / "agg_maps.joblib")
