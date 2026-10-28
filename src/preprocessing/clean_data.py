"""Column-level cleaning after merge: drop high-missing columns, fix dtypes, standardize missing markers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _drop_high_missing(
    df: pd.DataFrame,
    max_frac: float,
    protected: list[str],
) -> pd.DataFrame:
    """Drop columns whose fraction of missing values exceeds *max_frac*.

    Columns listed in *protected* are never dropped.
    """
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > max_frac].index.tolist()
    to_drop = [c for c in to_drop if c not in protected]

    if to_drop:
        logger.info(
            "Dropping %d columns with >%.0f%% missing: %s",
            len(to_drop),
            max_frac * 100,
            to_drop[:10],  # log first 10 to avoid clutter
        )
        df = df.drop(columns=to_drop)
    else:
        logger.info("No columns exceed %.0f%% missing threshold.", max_frac * 100)

    return df


def _force_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop explicitly listed useless columns."""
    existing = [c for c in cols if c in df.columns]
    if existing:
        logger.info("Force-dropping columns: %s", existing)
        df = df.drop(columns=existing)
    return df


def _standardise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common non-NaN missing markers with ``np.nan``."""
    placeholders = ["NotFound", "not_found", "", " ", "NA", "nan", "None", "-1"]
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        mask = df[col].isin(placeholders)
        if mask.any():
            df.loc[mask, col] = np.nan
    return df


def _cast_categories(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Cast specified columns to pandas ``category`` dtype."""
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _add_event_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add a datetime 'event_timestamp' column for Feast.
    
    TransactionDT is an offset in seconds. We use a fixed reference date.
    """
    if "TransactionDT" in df.columns:
        reference_date = pd.Timestamp("2017-12-01")
        df["event_timestamp"] = reference_date + pd.to_timedelta(df["TransactionDT"], unit="s")
    else:
        # Fallback for identity-only or tests
        df["event_timestamp"] = pd.Timestamp.now()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Merged (raw) DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with fewer columns and consistent dtypes.
    """
    data_cfg = load_config("data")
    clean_cfg = data_cfg.get("cleaning", {})

    max_missing = clean_cfg.get("max_missing_fraction", 0.50)
    protected = clean_cfg.get("protected_columns", [])
    force_drop = clean_cfg.get("force_drop_columns", [])
    cat_cols = clean_cfg.get("category_columns", [])

    n_cols_before = len(df.columns)
    logger.info("Cleaning started — %d columns", n_cols_before)

    df = _standardise_missing(df)
    df = _force_drop(df, force_drop)
    df = _drop_high_missing(df, max_missing, protected)
    df = _cast_categories(df, cat_cols)
    df = _add_event_timestamp(df)

    logger.info("Cleaning done — %d → %d columns", n_cols_before, len(df.columns))
    return df
