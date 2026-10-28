"""
src/data/split_data.py
──────────────────────
Chronological train / validation / test split on ``TransactionDT``.

LEAKAGE PREVENTION
──────────────────
• Rows are sorted by ``TransactionDT`` and split by position so
  train < valid < test in time.
• No ``TransactionID`` or time overlap between splits.
• After this step, imputers / encoders must be fit **only** on the
  training fold.
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SplitLeakageError(Exception):
    """Raised when a leakage check fails after splitting."""


def chronological_split(
    df: pd.DataFrame,
    time_col: str | None = None,
    train_frac: float | None = None,
    valid_frac: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically by the time column.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with the time column present.
    time_col : str, optional
        Column to sort by.  Defaults to ``configs/split.yaml → time_column``.
    train_frac, valid_frac : float, optional
        Override the fractions from config (test = 1 − train − valid).

    Returns
    -------
    (train_df, valid_df, test_df)

    Raises
    ------
    SplitLeakageError
        If any post-split leakage check fails.
    """
    split_cfg = load_config("split")

    time_col = time_col or split_cfg.get("time_column", "TransactionDT")
    train_frac = train_frac or split_cfg.get("train_fraction", 0.70)
    valid_frac = valid_frac or split_cfg.get("valid_fraction", 0.15)
    test_frac = round(1.0 - train_frac - valid_frac, 4)

    if test_frac <= 0:
        raise ValueError(
            f"train_frac ({train_frac}) + valid_frac ({valid_frac}) >= 1.0  — no room for test."
        )

    # Sort by time
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))

    train_df = df_sorted.iloc[:train_end].copy()
    valid_df = df_sorted.iloc[train_end:valid_end].copy()
    test_df = df_sorted.iloc[valid_end:].copy()

    logger.info(
        "Chronological split: train=%d (%.0f%%)  valid=%d (%.0f%%)  test=%d (%.0f%%)",
        len(train_df), 100 * len(train_df) / n,
        len(valid_df), 100 * len(valid_df) / n,
        len(test_df),  100 * len(test_df) / n,
    )

    # ── Safety checks ───────────────────────────────────────
    checks_cfg = split_cfg.get("checks", {})

    if checks_cfg.get("assert_no_time_overlap", True):
        _assert_no_time_overlap(train_df, valid_df, test_df, time_col)

    if checks_cfg.get("assert_no_id_overlap", True):
        _assert_no_id_overlap(train_df, valid_df, test_df)

    if checks_cfg.get("log_split_stats", True):
        _log_split_stats(train_df, valid_df, test_df, time_col)

    return train_df, valid_df, test_df


# ── Internal helpers ────────────────────────────────────────────


def _assert_no_time_overlap(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, time_col: str
) -> None:
    """Ensure max(train time) < min(valid time) < min(test time)."""
    train_max = train[time_col].max()
    valid_min = valid[time_col].min()
    valid_max = valid[time_col].max()
    test_min = test[time_col].min()

    if train_max >= valid_min:
        raise SplitLeakageError(
            f"Time overlap between train (max={train_max}) and valid (min={valid_min})."
        )
    if valid_max >= test_min:
        raise SplitLeakageError(
            f"Time overlap between valid (max={valid_max}) and test (min={test_min})."
        )
    logger.info("✓ No time overlap between splits.")


def _assert_no_id_overlap(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Ensure TransactionID sets are disjoint."""
    id_col = "TransactionID"
    if id_col not in train.columns:
        return

    train_ids = set(train[id_col])
    valid_ids = set(valid[id_col])
    test_ids = set(test[id_col])

    tv = train_ids & valid_ids
    tt = train_ids & test_ids
    vt = valid_ids & test_ids

    if tv or tt or vt:
        raise SplitLeakageError(
            f"TransactionID overlap — train∩valid={len(tv)}, "
            f"train∩test={len(tt)}, valid∩test={len(vt)}"
        )
    logger.info("✓ No TransactionID overlap between splits.")


def _log_split_stats(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, time_col: str
) -> None:
    """Log fraud rates and time ranges for each split."""
    target = "isFraud"
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        fraud_rate = split[target].mean() * 100 if target in split.columns else float("nan")
        t_min, t_max = split[time_col].min(), split[time_col].max()
        logger.info(
            "  %s — %d rows, fraud=%.2f%%, %s=[%s, %s]",
            name, len(split), fraud_rate, time_col, t_min, t_max,
        )
