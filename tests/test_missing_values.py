"""
tests/test_missing_values.py
─────────────────────────────
Tests for src/preprocessing/missing_values.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.missing_values import fit_imputers, transform_impute


class TestImputation:
    def test_no_nans_after_impute(self):
        train = pd.DataFrame({
            "num_a": [1.0, 2.0, np.nan, 4.0],
            "num_b": [10.0, np.nan, 30.0, 40.0],
            "cat_a": ["x", "y", np.nan, "x"],
        })
        imputers = fit_imputers(train)
        result = transform_impute(train, imputers)
        # Numeric columns should have no NaN
        assert result[["num_a", "num_b"]].isna().sum().sum() == 0

    def test_median_strategy(self):
        train = pd.DataFrame({"val": [1.0, 2.0, 3.0, np.nan]})
        imputers = fit_imputers(train, numeric_strategy="median")
        result = transform_impute(train, imputers)
        assert result["val"].iloc[3] == 2.0  # median of [1, 2, 3]

    def test_transform_handles_missing_trained_columns(self):
        train = pd.DataFrame({
            "num_a": [1.0, 2.0, np.nan, 4.0],
            "num_b": [10.0, np.nan, 30.0, 40.0],
            "cat_a": ["x", "y", np.nan, "x"],
        })
        infer = pd.DataFrame({
            "num_a": [5.0],
            "cat_a": ["x"],
        })

        imputers = fit_imputers(train)
        result = transform_impute(infer, imputers)

        assert {"num_a", "num_b", "cat_a"}.issubset(result.columns)
        assert result.loc[0, "num_b"] == 30.0
        assert result.loc[0, "cat_a"] == "x"
