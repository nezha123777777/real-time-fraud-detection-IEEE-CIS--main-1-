from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines import offline_scoring_pipeline as pipeline


class TestOfflineScoring:
    def test_supports_csv_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline,
            "load_config",
            lambda name: {
                "data": {"X_test": "data/processed/X_test.parquet"},
                "reports": {"tables_dir": "reports/tables"},
                "artifacts": {"feature_lists_dir": "artifacts/models/artifacts/feature_lists"},
                "models": {"final_dir": "artifacts/models/final"},
            },
        )
        monkeypatch.setattr(pipeline, "resolve_path", lambda value: tmp_path / value)
        monkeypatch.setattr(
            pipeline,
            "load_parquet",
            lambda path: pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}),
        )
        monkeypatch.setattr(pipeline, "load_expected_columns", lambda path: ["f1", "f2"])
        monkeypatch.setattr(pipeline, "align_expected_columns", lambda frame, cols: frame.loc[:, cols])
        monkeypatch.setattr(pipeline, "load_onnx_session", lambda path: object())
        monkeypatch.setattr(pipeline, "load_threshold", lambda: {"threshold": 0.5})
        monkeypatch.setattr(
            pipeline,
            "predict_proba_onnx",
            lambda model, frame: np.array([0.2, 0.9]),
        )

        def fail_save_parquet(*args, **kwargs):
            raise AssertionError("CSV output should not call save_parquet")

        monkeypatch.setattr(pipeline, "save_parquet", fail_save_parquet)

        output_path = tmp_path / "reports" / "tables" / "scores.csv"
        result = pipeline.run_offline_scoring(output_path=output_path)

        assert output_path.exists()
        assert list(result.columns) == ["fraud_prob", "fraud_label"]
        assert result["fraud_label"].tolist() == [0, 1]
