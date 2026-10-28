from __future__ import annotations

import pandas as pd

from deployment import runtime


class TestRuntimeRefactor:
    def test_explain_payload_reuses_preprocessed_features(self, monkeypatch):
        X = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        calls = {"preprocess": 0}

        def fake_preprocess(payload, artifacts):
            calls["preprocess"] += 1
            return X

        monkeypatch.setattr(runtime, "preprocess_payload", fake_preprocess)
        monkeypatch.setattr(runtime, "predict_proba_onnx", lambda model, frame: [0.9])
        monkeypatch.setattr(
            runtime,
            "score_anomalies",
            lambda model, frame: ([0.2], [True]),
        )
        monkeypatch.setattr(
            runtime,
            "explain_prediction",
            lambda model, frame: {
                "base_value": 0.1,
                "feature_count": len(frame.columns),
                "top_features": [],
            },
        )

        result = runtime.explain_payload(
            payload={"TransactionAmt": 99.0, "TransactionDT": 10},
            artifacts={
                "model": object(),
                "native_model": object(),
                "threshold": 0.5,
                "review_threshold": 0.35,
                "model_version": "xgb_final",
                "isolation_model": object(),
            },
        )

        assert calls["preprocess"] == 1
        assert result["fraud_probability"] == 0.9
        assert result["fraud_label"] == 1
        assert result["action"] == "BLOCK"
        assert result["anomaly_flag"] is True
        assert result["anomaly_score"] == 0.2
        assert result["feature_count"] == 2
