"""
src/utils/io.py
---------------
I/O helpers for reading / writing parquet and pickle artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to parquet, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved parquet (%d rows, %d cols) -> %s", len(df), len(df.columns), path)
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded parquet (%d rows, %d cols) <- %s", len(df), len(df.columns), path)
    return df


def save_artifact(obj: Any, path: str | Path) -> Path:
    """Persist a Python object via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artefact -> %s", path)
    return path


def load_artifact(path: str | Path) -> Any:
    """Load a joblib-pickled artefact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found: {path}")
    obj = joblib.load(path)
    logger.info("Loaded artefact <- %s", path)
    return obj
