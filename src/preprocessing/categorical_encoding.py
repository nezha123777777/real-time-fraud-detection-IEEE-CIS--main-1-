"""Categorical encoding helpers (frequency-based); fit on train only with save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import resolve_path, load_config
from src.utils.io import save_artifact, load_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrequencyEncoder:
    """Encode categories by their frequency in the training set.

    Parameters
    ----------
    normalize : bool
        If ``True`` (default), encode as fractions; otherwise as raw counts.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self.freq_maps: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "FrequencyEncoder":
        """Compute frequency maps from *training* data only."""
        for col in columns:
            if col not in df.columns:
                continue
            counts = df[col].value_counts(normalize=self.normalize)
            self.freq_maps[col] = counts.to_dict()
        logger.info("FrequencyEncoder fitted on %d columns.", len(self.freq_maps))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map categories to frequencies; unseen → 0."""
        df = df.copy()
        for col, fmap in self.freq_maps.items():
            if col in df.columns:
                df[col] = df[col].map(fmap).fillna(0).astype("float32")
        return df

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)


class SafeLabelEncoder:
    """Label encoder that handles unseen categories at transform time.

    Unseen values are mapped to ``-1``.
    """

    def __init__(self) -> None:
        self.mappings: dict[str, dict[Any, int]] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "SafeLabelEncoder":
        """Build label maps from *training* data only."""
        for col in columns:
            if col not in df.columns:
                continue
            unique_vals = df[col].dropna().unique()
            self.mappings[col] = {v: i for i, v in enumerate(sorted(unique_vals, key=str))}
        logger.info("SafeLabelEncoder fitted on %d columns.", len(self.mappings))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map to integer codes; unseen → ``-1``."""
        df = df.copy()
        for col, mapping in self.mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1).astype("int32")
        return df

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)


def save_encoders(encoders: dict[str, Any], directory: str | Path | None = None) -> Path:
    """Persist encoders to disk."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["artifacts"]["encoders_dir"])
    return save_artifact(encoders, Path(directory) / "encoders.joblib")


def load_encoders(directory: str | Path | None = None) -> dict[str, Any]:
    """Load previously saved encoders."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["artifacts"]["encoders_dir"])
    return load_artifact(Path(directory) / "encoders.joblib")
