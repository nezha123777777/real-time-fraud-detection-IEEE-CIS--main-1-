"""
tests/conftest.py
─────────────────
Shared pytest fixtures for the IEEE-CIS Fraud Detection test suite.

These fixtures create small synthetic DataFrames that mirror the
structure of the real dataset, enabling fast unit tests without
requiring the actual Kaggle CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def sample_transaction_df() -> pd.DataFrame:
    """Minimal transaction DataFrame (100 rows)."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "isFraud": rng.choice([0, 1], size=n, p=[0.965, 0.035]),
        "TransactionDT": np.sort(rng.randint(86400, 86400 * 180, size=n)),
        "TransactionAmt": rng.exponential(100, size=n).round(2),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=n),
        "card1": rng.randint(1000, 9000, size=n),
        "card4": rng.choice(["visa", "mastercard", "discover", np.nan], size=n),
        "card6": rng.choice(["debit", "credit", np.nan], size=n),
        "addr1": rng.choice([100, 200, 300, 400, np.nan], size=n),
        "D1": rng.choice([0, 1, 5, 14, 30, np.nan], size=n),
        "M1": rng.choice(["T", "F", np.nan], size=n),
    })


@pytest.fixture()
def sample_identity_df() -> pd.DataFrame:
    """Minimal identity DataFrame (50 rows — not all transactions have identity)."""
    rng = np.random.RandomState(43)
    n = 50
    return pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),  # only first 50 TXNs
        "id_31": rng.choice(["chrome 70.0", "firefox 65.0", "safari mobile", "samsung browser", np.nan], size=n),
        "id_33": rng.choice(["1920x1080", "1366x768", "375x667", np.nan], size=n),
        "DeviceType": rng.choice(["desktop", "mobile", np.nan], size=n),
    })


@pytest.fixture()
def merged_df(sample_transaction_df, sample_identity_df) -> pd.DataFrame:
    """Merged transaction + identity (LEFT join)."""
    return sample_transaction_df.merge(sample_identity_df, on="TransactionID", how="left")
