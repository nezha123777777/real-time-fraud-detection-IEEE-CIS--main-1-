"""Time-based features derived from TransactionDT (hour, day, weekend, night, elapsed days, card velocity)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Seconds in a day — TransactionDT is in seconds.
_SECONDS_PER_DAY = 86_400


def add_time_features(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    """Create time-derived features from *time_col*.

    New columns
    -----------
    ``hour``          — inferred hour of day (0–23)
    ``day_of_week``   — inferred day of week (0–6, Mon=0)
    ``elapsed_days``  — total elapsed days since reference
    ``is_weekend``    — 1 if Saturday or Sunday
    ``is_night``      — 1 if hour ∈ [0, 6)

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str

    Returns
    -------
    pd.DataFrame  with new columns added.
    """
    df = df.copy()

    dt = df[time_col].astype("float64")

    # Elapsed days (continuous)
    df["elapsed_days"] = dt / _SECONDS_PER_DAY

    # Inferred hour & day
    # We assume the reference point starts on a Monday at midnight.
    total_seconds_in_day = dt % _SECONDS_PER_DAY
    df["hour"] = (total_seconds_in_day // 3600).astype("int8")
    df["day_of_week"] = ((dt // _SECONDS_PER_DAY) % 7).astype("int8")

    # Binary flags
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype("int8")

    logger.info(
        "Added time features: hour, day_of_week, elapsed_days, is_weekend, is_night"
    )
    return df


def add_card_velocity(
    df: pd.DataFrame,
    card_col: str = "card1",
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """Compute time-since-last-transaction for each card.

    This is a **backward-looking** feature — it only uses past
    information relative to the current row, so it does **not** create
    leakage even when computed on the full dataset.

    New column: ``time_since_last_{card_col}``
    """
    df = df.copy()
    df = df.sort_values(time_col)

    col_name = f"time_since_last_{card_col}"
    df[col_name] = df.groupby(card_col)[time_col].diff().astype("float32")
    # First occurrence for each card → NaN, fill with −1 sentinel
    df[col_name] = df[col_name].fillna(-1)

    logger.info("Added card velocity feature: %s", col_name)
    return df
