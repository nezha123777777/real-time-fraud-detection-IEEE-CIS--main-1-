"""
tests/test_clean_data.py
────────────────────────
Tests for src/preprocessing/clean_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.clean_data import (
    _drop_high_missing,
    _standardise_missing,
    clean_data,
)


class TestDropHighMissing:
    def test_drops_columns_above_threshold(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, np.nan, np.nan, np.nan, np.nan],  # 80% missing
            "c": [1, 2, np.nan, 4, 5],  # 20% missing
        })
        result = _drop_high_missing(df, max_frac=0.5, protected=[])
        assert "a" in result.columns
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_protected_columns_kept(self):
        df = pd.DataFrame({
            "isFraud": [np.nan] * 5,  # 100% missing but protected
            "junk": [np.nan] * 5,
        })
        result = _drop_high_missing(df, max_frac=0.5, protected=["isFraud"])
        assert "isFraud" in result.columns
        assert "junk" not in result.columns


class TestStandardiseMissing:
    def test_replaces_placeholder_strings(self):
        df = pd.DataFrame({"col": ["hello", "NotFound", "", "valid"]})
        result = _standardise_missing(df)
        assert result["col"].isna().sum() == 2  # "NotFound" and ""
