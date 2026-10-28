"""
tests/test_load_data.py
───────────────────────
Tests for src/data/load_data.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_data import load_raw_csv


class TestLoadRawCsv:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_csv(tmp_path / "nonexistent.csv")

    def test_loads_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        df = load_raw_csv(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_nrows_limit(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": range(100)}).to_csv(csv_path, index=False)
        df = load_raw_csv(csv_path, nrows=10)
        assert len(df) == 10
