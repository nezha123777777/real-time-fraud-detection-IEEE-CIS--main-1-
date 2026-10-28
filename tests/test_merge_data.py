"""
tests/test_merge_data.py
────────────────────────
Tests for src/data/merge_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.merge_data import merge_transaction_identity


class TestMergeTransactionIdentity:
    """Verify the LEFT JOIN merge behaviour."""

    def test_row_count_preserved(self, sample_transaction_df, sample_identity_df):
        """LEFT join must keep all transaction rows."""
        merged = merge_transaction_identity(
            sample_transaction_df, sample_identity_df,
            join_key="TransactionID", join_how="left",
        )
        assert len(merged) == len(sample_transaction_df)

    def test_columns_from_both_tables(self, sample_transaction_df, sample_identity_df):
        """Merged df should contain columns from both tables."""
        merged = merge_transaction_identity(
            sample_transaction_df, sample_identity_df,
            join_key="TransactionID", join_how="left",
        )
        for col in sample_transaction_df.columns:
            assert col in merged.columns
        for col in sample_identity_df.columns:
            assert col in merged.columns

    def test_identity_nans_for_unmatched(self, sample_transaction_df, sample_identity_df):
        """Transactions without identity data should have NaN in identity cols."""
        merged = merge_transaction_identity(
            sample_transaction_df, sample_identity_df,
            join_key="TransactionID", join_how="left",
        )
        # IDs 51-100 have no identity data → identity columns should be NaN
        unmatched = merged[merged["TransactionID"] > 50]
        id_cols = [c for c in sample_identity_df.columns if c != "TransactionID"]
        for col in id_cols:
            assert unmatched[col].isna().all(), f"Expected NaN for unmatched '{col}'"

    def test_target_column_preserved(self, sample_transaction_df, sample_identity_df):
        """isFraud must not be dropped or altered by the merge."""
        merged = merge_transaction_identity(
            sample_transaction_df, sample_identity_df,
            join_key="TransactionID", join_how="left",
        )
        assert "isFraud" in merged.columns
        assert merged["isFraud"].isna().sum() == 0

    def test_missing_join_key_raises(self, sample_transaction_df):
        """Should raise if join key is missing in either table."""
        bad_identity = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValueError, match="Join key"):
            merge_transaction_identity(
                sample_transaction_df, bad_identity,
                join_key="TransactionID", join_how="left",
            )

    def test_duplicate_identity_raises(self, sample_transaction_df):
        """Duplicate TransactionIDs in identity should cause a row-count mismatch."""
        dup_identity = pd.DataFrame({
            "TransactionID": [1, 1, 2],
            "id_31": ["chrome", "firefox", "safari"],
        })
        with pytest.raises(ValueError, match="LEFT join produced"):
            merge_transaction_identity(
                sample_transaction_df, dup_identity,
                join_key="TransactionID", join_how="left",
            )
