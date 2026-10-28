from __future__ import annotations

import pandas as pd
import pytest

from src.pipelines.training_pipeline import build_clean_holdout_replay
from src.utils.io import save_parquet


class TestBuildCleanHoldoutReplay:
    def test_filters_to_holdout_rows_and_sorts_by_time(self, tmp_path):
        cleaned_path = tmp_path / "cleaned_train.parquet"
        cleaned_df = pd.DataFrame({
            "TransactionID": [30, 10, 20],
            "TransactionDT": [300, 100, 200],
            "TransactionAmt": [3.0, 1.0, 2.0],
            "isFraud": [0, 1, 0],
        })
        save_parquet(cleaned_df, cleaned_path)

        replay_df = build_clean_holdout_replay(
            cleaned_path,
            pd.Series([20, 10]),
        )

        assert replay_df["TransactionID"].tolist() == [10, 20]
        assert replay_df["TransactionDT"].tolist() == [100, 200]
        assert replay_df["isFraud"].tolist() == [1, 0]

    def test_raises_when_holdout_ids_are_missing(self, tmp_path):
        cleaned_path = tmp_path / "cleaned_train.parquet"
        save_parquet(
            pd.DataFrame({
                "TransactionID": [10],
                "TransactionDT": [100],
                "TransactionAmt": [1.0],
            }),
            cleaned_path,
        )

        with pytest.raises(ValueError, match="missing 1 held-out TransactionID"):
            build_clean_holdout_replay(cleaned_path, pd.Series([10, 20]))
