"""
tests/test_feature_engineering.py
──────────────────────────────────
Tests for feature engineering modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.time_features import add_time_features, add_card_velocity
from src.features.identity_features import add_browser_features, add_screen_resolution_features
from src.features.build_features import build_features_stateless, _add_amount_features
from src.features.aggregation_features import fit_aggregations, transform_aggregations


class TestTimeFeatures:
    """Verify time-based feature engineering."""

    def test_expected_columns_created(self, merged_df):
        result = add_time_features(merged_df)
        for col in ["hour", "day_of_week", "elapsed_days", "is_weekend", "is_night"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_hour_range(self, merged_df):
        result = add_time_features(merged_df)
        assert result["hour"].between(0, 23).all()

    def test_day_of_week_range(self, merged_df):
        result = add_time_features(merged_df)
        assert result["day_of_week"].between(0, 6).all()

    def test_elapsed_days_positive(self, merged_df):
        result = add_time_features(merged_df)
        assert (result["elapsed_days"] > 0).all()

    def test_is_weekend_binary(self, merged_df):
        result = add_time_features(merged_df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})


class TestCardVelocity:
    """Verify card velocity (time-since-last) feature."""

    def test_column_created(self, merged_df):
        result = add_card_velocity(merged_df, card_col="card1")
        assert "time_since_last_card1" in result.columns

    def test_first_occurrence_is_sentinel(self, merged_df):
        """First transaction per card should have -1 (sentinel)."""
        result = add_card_velocity(merged_df, card_col="card1")
        result_sorted = result.sort_values("TransactionDT")
        first_per_card = result_sorted.groupby("card1").first()
        assert (first_per_card["time_since_last_card1"] == -1).all()

    def test_no_negative_deltas(self, merged_df):
        """Time deltas (excluding sentinel -1) should be >= 0."""
        result = add_card_velocity(merged_df, card_col="card1")
        non_sentinel = result.loc[result["time_since_last_card1"] != -1, "time_since_last_card1"]
        if len(non_sentinel) > 0:
            assert (non_sentinel >= 0).all()


class TestBrowserFeatures:
    """Verify browser feature extraction."""

    def test_browser_family_column(self, merged_df):
        result = add_browser_features(merged_df)
        assert "browser_family" in result.columns

    def test_known_families_extracted(self):
        df = pd.DataFrame({"id_31": ["chrome 70.0", "firefox 65", "safari mobile"]})
        result = add_browser_features(df)
        assert list(result["browser_family"]) == ["chrome", "firefox", "safari"]

    def test_mobile_flag(self):
        df = pd.DataFrame({"id_31": ["safari mobile", "chrome 70.0"]})
        result = add_browser_features(df)
        assert result["is_mobile_browser"].tolist() == [1, 0]


class TestScreenResolution:
    """Verify screen resolution parsing."""

    def test_parsed_correctly(self):
        df = pd.DataFrame({"id_33": ["1920x1080", "1366x768"]})
        result = add_screen_resolution_features(df)
        assert result["screen_width"].tolist() == [1920.0, 1366.0]
        assert result["screen_height"].tolist() == [1080.0, 768.0]
        assert result["screen_area"].tolist() == [1920 * 1080, 1366 * 768]


class TestAmountFeatures:
    """Verify TransactionAmt-derived features."""

    def test_log_amount_no_inf(self, merged_df):
        result = _add_amount_features(merged_df)
        assert "log_TransactionAmt" in result.columns
        assert not np.isinf(result["log_TransactionAmt"]).any()

    def test_round_amount_flag(self):
        df = pd.DataFrame({"TransactionAmt": [100.0, 99.99, 50.0, 25.50]})
        result = _add_amount_features(df)
        assert result["amt_is_round"].tolist() == [1, 0, 1, 0]


class TestAggregationFeatures:
    """Verify aggregation features respect train-only fitting."""

    def test_fit_transform_creates_columns(self, merged_df):
        agg_maps = fit_aggregations(merged_df)
        result = transform_aggregations(merged_df, agg_maps)
        # At least one aggregation column should exist
        agg_cols = [c for c in result.columns if c.startswith("card1_") or c.startswith("addr1_")]
        assert len(agg_cols) > 0

    def test_unseen_groups_get_fill(self, merged_df):
        """Groups unseen in train should be filled, not produce NaN."""
        agg_maps = fit_aggregations(merged_df)

        # Create a test df with a new card1 value
        test_df = merged_df.head(5).copy()
        test_df["card1"] = 99999  # unseen value

        result = transform_aggregations(test_df, agg_maps)
        agg_cols = [c for c in result.columns if c.startswith("card1_TransactionAmt")]
        for col in agg_cols:
            assert not result[col].isna().any(), f"NaN found in '{col}' for unseen group"

    def test_no_infinite_values(self, merged_df):
        agg_maps = fit_aggregations(merged_df)
        result = transform_aggregations(merged_df, agg_maps)
        num_cols = result.select_dtypes(include="number").columns
        assert not np.isinf(result[num_cols]).any().any()
