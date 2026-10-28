"""
src/data/validate_data.py
─────────────────────────
Post-merge validation checks driven by ``configs/data.yaml``.

These checks run early in the pipeline to catch schema or quality issues
before expensive processing begins.
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when a validation check fails."""


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Assert that all required columns are present.

    Raises
    ------
    DataValidationError
        If any column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    logger.info("✓ All %d required columns present.", len(required))


def validate_target(df: pd.DataFrame, target_col: str, valid_values: list[int]) -> None:
    """Assert that the target column is binary with expected values.

    Raises
    ------
    DataValidationError
        If the target contains unexpected values or is entirely null.
    """
    if target_col not in df.columns:
        raise DataValidationError(f"Target column '{target_col}' not found.")

    if df[target_col].isna().all():
        raise DataValidationError(f"Target column '{target_col}' is entirely null.")

    unexpected = set(df[target_col].dropna().unique()) - set(valid_values)
    if unexpected:
        raise DataValidationError(
            f"Target column '{target_col}' has unexpected values: {unexpected}"
        )

    fraud_rate = df[target_col].mean()
    logger.info(
        "✓ Target '%s' is valid — fraud rate: %.2f%% (%d / %d)",
        target_col,
        fraud_rate * 100,
        int(df[target_col].sum()),
        len(df),
    )


def validate_unique_id(df: pd.DataFrame, id_col: str) -> None:
    """Assert that the ID column has no duplicates.

    Raises
    ------
    DataValidationError
        If duplicates are found.
    """
    n_dup = df[id_col].duplicated().sum()
    if n_dup > 0:
        raise DataValidationError(
            f"Column '{id_col}' has {n_dup} duplicate values."
        )
    logger.info("✓ ID column '%s' is unique (%d rows).", id_col, len(df))


def validate_merged_data(df: pd.DataFrame) -> None:
    """Run the full suite of validation checks on the merged DataFrame.

    Reads expected rules from ``configs/data.yaml → validation``.

    Parameters
    ----------
    df : pd.DataFrame
        Merged transaction + identity DataFrame.

    Raises
    ------
    DataValidationError
        On the first check that fails.
    """
    data_cfg = load_config("data")
    val_cfg = data_cfg.get("validation", {})

    required_cols = val_cfg.get("required_columns", [])
    target_col = val_cfg.get("target_column", "isFraud")
    target_vals = val_cfg.get("target_values", [0, 1])
    unique_id_col = val_cfg.get("unique_id_column", "TransactionID")

    logger.info("Running data validation checks …")
    validate_required_columns(df, required_cols)
    validate_target(df, target_col, target_vals)
    validate_unique_id(df, unique_id_col)
    logger.info("All validation checks passed.")
