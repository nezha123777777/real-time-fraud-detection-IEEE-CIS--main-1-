"""Optional numeric scaling / log transforms; fit on training data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import resolve_path, load_config
from src.utils.io import save_artifact, load_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


def log1p_transform(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Apply ``log1p`` to skewed numeric columns (e.g. TransactionAmt).

    This is a stateless transform — no fitting required — so it is safe
    to apply before or after the split.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    logger.info("Applied log1p to columns: %s", columns)
    return df


def fit_scaler(
    df: pd.DataFrame, columns: list[str]
) -> tuple[StandardScaler, list[str]]:
    """Fit a ``StandardScaler`` on training numeric columns.

    Returns
    -------
    (fitted_scaler, column_list)
    """
    cols = [c for c in columns if c in df.columns]
    scaler = StandardScaler()
    scaler.fit(df[cols])
    logger.info("Fitted StandardScaler on %d columns.", len(cols))
    return scaler, cols


def transform_scale(
    df: pd.DataFrame,
    scaler: StandardScaler,
    columns: list[str],
) -> pd.DataFrame:
    """Apply a pre-fitted scaler."""
    df = df.copy()
    cols = [c for c in columns if c in df.columns]
    df[cols] = scaler.transform(df[cols])
    return df
