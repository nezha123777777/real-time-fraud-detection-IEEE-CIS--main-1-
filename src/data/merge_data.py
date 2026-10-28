"""
src/data/merge_data.py
──────────────────────
Merge transaction and identity tables on ``TransactionID``.

Uses a **LEFT JOIN** so that every transaction is preserved even when
the identity record is missing (which is common — identity data is only
available for a subset of transactions).
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def merge_transaction_identity(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    join_key: str | None = None,
    join_how: str | None = None,
) -> pd.DataFrame:
    """Merge transaction and identity DataFrames.

    Parameters
    ----------
    transaction_df : pd.DataFrame
        Transaction-level data (one row per ``TransactionID``).
    identity_df : pd.DataFrame
        Identity / device data (may be missing for some transactions).
    join_key : str, optional
        Column to join on. Defaults to ``configs/data.yaml → merging.join_key``.
    join_how : str, optional
        Join type (``"left"``, ``"inner"``, …). Defaults to config value.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.  For a LEFT join the row count should equal
        ``len(transaction_df)``.

    Raises
    ------
    ValueError
        If the join key is missing in either DataFrame, or if the
        resulting row count is unexpected.
    """
    data_cfg = load_config("data")
    merging = data_cfg.get("merging", {})

    join_key = join_key or merging.get("join_key", "TransactionID")
    join_how = join_how or merging.get("join_how", "left")

    # ── Pre-checks ──────────────────────────────────────────
    if join_key not in transaction_df.columns:
        raise ValueError(f"Join key '{join_key}' not found in transaction_df columns.")
    if join_key not in identity_df.columns:
        raise ValueError(f"Join key '{join_key}' not found in identity_df columns.")

    n_txn = len(transaction_df)
    logger.info(
        "Merging transaction (%d rows) + identity (%d rows) on '%s' [%s join]",
        n_txn,
        len(identity_df),
        join_key,
        join_how,
    )

    merged = transaction_df.merge(identity_df, on=join_key, how=join_how)

    # ── Post-checks ─────────────────────────────────────────
    if join_how == "left" and len(merged) != n_txn:
        raise ValueError(
            f"LEFT join produced {len(merged)} rows but expected {n_txn}. "
            "Check for duplicate TransactionIDs in the identity table."
        )

    n_identity_matched = merged[identity_df.columns[1]].notna().sum() if len(identity_df.columns) > 1 else 0
    logger.info(
        "  → merged shape: %s  |  identity match rate: %.1f%%",
        merged.shape,
        100 * n_identity_matched / len(merged),
    )

    return merged
