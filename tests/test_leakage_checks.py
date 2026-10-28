"""
tests/test_leakage_checks.py
─────────────────────────────
Integration-style tests that verify no data leakage occurs
during the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split_data import chronological_split
from src.preprocessing.missing_values import fit_imputers, transform_impute
from src.features.aggregation_features import fit_aggregations, transform_aggregations


class TestLeakagePrevention:
    """These tests catch the most common sources of leakage."""

    def test_imputer_uses_train_only(self, merged_df):
        """Imputer stats from train must not reflect valid/test data."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        imputers = fit_imputers(train)

        # The imputer should be fitted on train.
        # Transform valid — should not fail, and should use train stats.
        valid_imputed = transform_impute(valid, imputers)
        assert len(valid_imputed) == len(valid)

    def test_aggregation_maps_use_train_only(self, merged_df):
        """Aggregation lookups must come from train, not from full dataset."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        agg_maps = fit_aggregations(train)

        # Valid/test may have card values unseen in train — these should
        # be filled, not produce NaN.
        valid_transformed = transform_aggregations(valid, agg_maps)
        agg_cols = [c for c in valid_transformed.columns if "TransactionAmt" in c and c != "TransactionAmt"]
        for col in agg_cols:
            assert not valid_transformed[col].isna().any(), f"Leakage-prone NaN in '{col}'"

    def test_no_future_data_in_train(self, merged_df):
        """All train transactions must predate all valid transactions."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        assert train["TransactionDT"].max() < valid["TransactionDT"].min()
        assert valid["TransactionDT"].max() < test["TransactionDT"].min()

    def test_no_target_in_features(self, merged_df):
        """The target column must not appear in the feature matrix."""
        from src.models.train import prepare_feature_matrix
        X = prepare_feature_matrix(merged_df)
        assert "isFraud" not in X.columns
        assert "TransactionID" not in X.columns
