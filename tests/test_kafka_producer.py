from __future__ import annotations

import pandas as pd
import pytest

from deployment.kafka import producer
from src.utils.io import save_parquet


class TestResolveSourcePath:
    def test_prefers_holdout_replay_when_present(self, tmp_path, monkeypatch):
        holdout_path = tmp_path / "cleaned_holdout_replay.parquet"
        legacy_path = tmp_path / "cleaned_train.parquet"
        save_parquet(pd.DataFrame({"TransactionID": [1]}), holdout_path)
        save_parquet(pd.DataFrame({"TransactionID": [2]}), legacy_path)

        monkeypatch.setattr(producer, "DEFAULT_SOURCE", holdout_path)
        monkeypatch.setattr(producer, "LEGACY_SOURCE", legacy_path)

        assert producer.resolve_source_path(str(holdout_path)) == holdout_path

    def test_falls_back_to_legacy_source(self, tmp_path, monkeypatch):
        holdout_path = tmp_path / "cleaned_holdout_replay.parquet"
        legacy_path = tmp_path / "cleaned_train.parquet"
        save_parquet(pd.DataFrame({"TransactionID": [2]}), legacy_path)

        monkeypatch.setattr(producer, "DEFAULT_SOURCE", holdout_path)
        monkeypatch.setattr(producer, "LEGACY_SOURCE", legacy_path)

        assert producer.resolve_source_path(str(holdout_path)) == legacy_path

    def test_raises_when_no_replay_source_exists(self, tmp_path, monkeypatch):
        holdout_path = tmp_path / "cleaned_holdout_replay.parquet"
        legacy_path = tmp_path / "cleaned_train.parquet"

        monkeypatch.setattr(producer, "DEFAULT_SOURCE", holdout_path)
        monkeypatch.setattr(producer, "LEGACY_SOURCE", legacy_path)

        with pytest.raises(FileNotFoundError, match="Source data not found"):
            producer.resolve_source_path(str(holdout_path))


class TestIterPayloads:
    def test_drops_label_and_adds_feed_name(self):
        df = pd.DataFrame({
            "TransactionID": [1],
            "TransactionDT": [10],
            "TransactionAmt": [99.5],
            "isFraud": [1],
        })

        payload = next(iter(producer.iter_payloads(df, limit=1, feed_name="holdout")))

        assert payload["feed_name"] == "holdout"
        assert payload["TransactionID"] == 1
        assert payload["TransactionAmt"] == 99.5
        assert "isFraud" not in payload
