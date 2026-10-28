"""Orchestrates feature engineering. Stateless helpers (time, identity, amount) plus stateful aggregation maps that must be fit on training data only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.time_features import add_time_features, add_card_velocity
from src.features.identity_features import add_identity_features
from src.features.aggregation_features import (
    fit_aggregations,
    transform_aggregations,
)
from src.features.feature_selection import drop_constant_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Stateless (safe before split) ──────────────────────────────


def build_features_stateless(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transforms that require no fitting.

    Safe to call on the full dataset before splitting because these
    transforms only use information from the current row or look
    backward in time.

    Transforms
    ----------
    1. Time features (hour, day_of_week, elapsed_days, …)
    2. Card velocity (time since last txn per card1)
    3. Identity / device features (browser, screen resolution)
    4. Transaction amount helpers (log amount, amount cents, amount bins)
    5. D-column helpers
    """
    logger.info("Building stateless features …")

    df = add_time_features(df)
    df = add_card_velocity(df)
    df = add_identity_features(df)
    df = _add_amount_features(df)
    df = _add_d_column_features(df)

    logger.info("Stateless features done — new shape: %s", df.shape)
    return df


# ── Stateful (must be fit on train only) ───────────────────────


def fit_stateful_features(train_df: pd.DataFrame) -> dict:
    """Fit aggregation-based features on the training set.

    Returns a dict of feature maps that can be serialised and re-used
    on validation / test sets.
    """
    logger.info("Fitting stateful (aggregation) features on training data …")
    agg_maps = fit_aggregations(train_df)
    return {"agg_maps": agg_maps}


def transform_stateful_features(df: pd.DataFrame, feature_maps: dict) -> pd.DataFrame:
    """Apply pre-fitted feature maps to any split."""
    agg_maps = feature_maps.get("agg_maps", {})
    if agg_maps:
        df = transform_aggregations(df, agg_maps)
    return df


# ── Private helpers ────────────────────────────────────────────


def _add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from ``TransactionAmt``."""
    df = df.copy()
    amt = df["TransactionAmt"]

    # Log-transformed amount
    df["log_TransactionAmt"] = np.log1p(amt.clip(lower=0)).astype("float32")

    # Decimal / cents portion (fraud transactions sometimes have round amounts)
    df["amt_decimal"] = (amt - amt.astype(int)).astype("float32")

    # Binary: is the amount a round number?
    df["amt_is_round"] = (df["amt_decimal"] == 0.0).astype("int8")

    logger.info("Added amount features: log_TransactionAmt, amt_decimal, amt_is_round")
    return df


def _add_d_column_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from D-columns (timedelta columns D1–D15).

    D1 is commonly interpreted as "days since last activity" and is
    one of the strongest single features in the IEEE-CIS dataset.
    """
    df = df.copy()

    # D1 — ratio to elapsed days (normalisation)
    if "D1" in df.columns:
        df["D1_to_elapsed"] = (
            df["D1"] / df["elapsed_days"].replace(0, np.nan)
        ).astype("float32")

    # Count of non-null D-columns per row (device fingerprint completeness)
    d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
    if d_cols:
        df["D_count_notna"] = df[d_cols].notna().sum(axis=1).astype("int8")

    logger.info("Added D-column features (D1_to_elapsed, D_count_notna).")
    return df
