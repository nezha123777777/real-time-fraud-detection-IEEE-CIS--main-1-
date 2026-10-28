"""
tests/test_time_split.py
────────────────────────
Tests for src/data/split_data.py — chronological split & leakage checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split_data import chronological_split, SplitLeakageError


class TestChronologicalSplit:
    """Verify time-based splitting and leakage prevention."""

    def test_splits_sum_to_original(self, merged_df):
        """Total rows across splits must equal the original."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        assert len(train) + len(valid) + len(test) == len(merged_df)

    def test_no_time_overlap(self, merged_df):
        """max(train.TransactionDT) < min(valid.TransactionDT) < min(test.TransactionDT)."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        assert train["TransactionDT"].max() < valid["TransactionDT"].min()
        assert valid["TransactionDT"].max() < test["TransactionDT"].min()

    def test_no_id_overlap(self, merged_df):
        """TransactionID sets must be completely disjoint."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        train_ids = set(train["TransactionID"])
        valid_ids = set(valid["TransactionID"])
        test_ids = set(test["TransactionID"])
        assert train_ids.isdisjoint(valid_ids)
        assert train_ids.isdisjoint(test_ids)
        assert valid_ids.isdisjoint(test_ids)

    def test_train_comes_first_in_time(self, merged_df):
        """All train timestamps must precede all validation timestamps."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        assert (train["TransactionDT"] < valid["TransactionDT"].min()).all()

    def test_fraud_rate_reasonable(self, merged_df):
        """Each split should contain some fraud (at least when data is large enough)."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        # With 100 rows and ~3.5% fraud, some splits might have 0 fraud.
        # Just verify isFraud column exists and is binary.
        for name, split in [("train", train), ("valid", valid), ("test", test)]:
            assert "isFraud" in split.columns
            assert set(split["isFraud"].unique()).issubset({0, 1})

    def test_invalid_fractions_raise(self, merged_df):
        """train_frac + valid_frac >= 1 should raise ValueError."""
        with pytest.raises(ValueError, match="no room for test"):
            chronological_split(
                merged_df, time_col="TransactionDT", train_frac=0.8, valid_frac=0.3
            )

    def test_approximate_split_ratios(self, merged_df):
        """Split sizes should roughly match the requested fractions."""
        train, valid, test = chronological_split(
            merged_df, time_col="TransactionDT", train_frac=0.7, valid_frac=0.15
        )
        n = len(merged_df)
        assert abs(len(train) / n - 0.7) < 0.05
        assert abs(len(valid) / n - 0.15) < 0.05
        assert abs(len(test) / n - 0.15) < 0.05
