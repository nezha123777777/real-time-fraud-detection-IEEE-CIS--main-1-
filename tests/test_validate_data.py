"""
tests/test_validate_data.py
───────────────────────────
Tests for src/data/validate_data.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate_data import (
    DataValidationError,
    validate_required_columns,
    validate_target,
    validate_unique_id,
)


class TestValidateRequiredColumns:
    def test_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        validate_required_columns(df, ["a", "b"])  # should not raise

    def test_fails_when_missing(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_required_columns(df, ["a", "b"])


class TestValidateTarget:
    def test_valid_binary_target(self):
        df = pd.DataFrame({"isFraud": [0, 1, 0, 1]})
        validate_target(df, "isFraud", [0, 1])

    def test_unexpected_values(self):
        df = pd.DataFrame({"isFraud": [0, 1, 2]})
        with pytest.raises(DataValidationError, match="unexpected values"):
            validate_target(df, "isFraud", [0, 1])


class TestValidateUniqueId:
    def test_unique_ids_pass(self):
        df = pd.DataFrame({"TransactionID": [1, 2, 3]})
        validate_unique_id(df, "TransactionID")

    def test_duplicate_ids_fail(self):
        df = pd.DataFrame({"TransactionID": [1, 1, 2]})
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_unique_id(df, "TransactionID")
