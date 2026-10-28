"""Load raw IEEE-CIS training CSV files into DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _apply_dtype_overrides(
    df: pd.DataFrame, overrides: dict[str, str] | None
) -> pd.DataFrame:
    """Cast columns to specified dtypes to reduce memory footprint."""
    if not overrides:
        return df
    for col, dtype in overrides.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                logger.warning("Could not cast column '%s' to %s — skipping.", col, dtype)
    return df


def load_raw_csv(
    path: str | Path,
    nrows: int | None = None,
    usecols: list[str] | None = None,
    dtype_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read a single CSV file with optional row / column limits.

    Parameters
    ----------
    path : str or Path
        Absolute or project-relative path to the CSV.
    nrows : int, optional
        Limit the number of rows read (useful for quick-dev runs).
    usecols : list[str], optional
        Subset of columns to read.
    dtype_overrides : dict, optional
        Column → dtype mapping applied after loading.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    logger.info("Loading CSV: %s (nrows=%s)", path.name, nrows)
    df = pd.read_csv(path, nrows=nrows, usecols=usecols)
    df = _apply_dtype_overrides(df, dtype_overrides)
    logger.info("  → shape: %s  |  memory: %.1f MB", df.shape, df.memory_usage(deep=True).sum() / 1e6)
    return df


def load_train_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``train_transaction.csv`` and ``train_identity.csv``.

    Reads configuration from ``configs/paths.yaml`` and
    ``configs/data.yaml``.

    Returns
    -------
    (transaction_df, identity_df)
    """
    paths_cfg = load_config("paths")
    data_cfg = load_config("data")
    loading = data_cfg.get("loading", {})

    nrows = loading.get("nrows")
    usecols = loading.get("usecols")
    dtype_overrides = loading.get("dtype_overrides")

    txn_path = resolve_path(paths_cfg["data"]["train_transaction"])
    idn_path = resolve_path(paths_cfg["data"]["train_identity"])

    txn_df = load_raw_csv(txn_path, nrows=nrows, usecols=usecols, dtype_overrides=dtype_overrides)
    idn_df = load_raw_csv(idn_path, nrows=nrows, usecols=usecols, dtype_overrides=dtype_overrides)

    return txn_df, idn_df
