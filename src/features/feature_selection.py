"""Helpers for dropping constant or duplicate columns."""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that have a single unique value (zero variance)."""
    nunique = df.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        logger.info("Dropping %d constant columns: %s", len(constant), constant[:10])
        df = df.drop(columns=constant)
    return df


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are exact duplicates of another column."""
    seen: dict[int, str] = {}
    to_drop: list[str] = []
    for col in df.columns:
        col_hash = hash(df[col].values.tobytes())
        if col_hash in seen:
            to_drop.append(col)
            logger.debug("Column '%s' duplicates '%s'.", col, seen[col_hash])
        else:
            seen[col_hash] = col

    if to_drop:
        logger.info("Dropping %d duplicate columns.", len(to_drop))
        df = df.drop(columns=to_drop)
    return df
