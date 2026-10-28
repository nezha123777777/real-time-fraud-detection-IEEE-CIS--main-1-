from __future__ import annotations

import pytest

from deployment.api import main


def _fake_artifacts() -> dict[str, object]:
    return {
        "model": object(),
        "threshold": 0.5,
    }


class TestApiRefactor:
    @pytest.mark.asyncio
    async def test_predict_endpoint_uses_shared_response_fields(self, monkeypatch):
        monkeypatch.setattr(main, "ARTIFACTS", _fake_artifacts())
        monkeypatch.setattr(
            main,
            "predict_payload",
            lambda payload, artifacts: {
                "fraud_probability": 0.42,
                "fraud_label": 0,
                "action": "ALLOW",
                "threshold": 0.5,
                "review_threshold": 0.35,
                "anomaly_flag": False,
                "anomaly_score": None,
                "model_version": "xgb_final",
            },
        )

        response = await main.predict_fraud(
            main.FraudPredictionRequest(TransactionAmt=150.5, TransactionDT=86450)
        )

        assert response.fraud_probability == 0.42
        assert response.fraud_label == 0
        assert response.action == "ALLOW"
        assert response.anomaly_score is None
        assert response.model_version == "xgb_final"
        assert isinstance(response.TransactionID, str)

    @pytest.mark.asyncio
    async def test_explain_endpoint_uses_shared_response_fields(self, monkeypatch):
        monkeypatch.setattr(main, "ARTIFACTS", _fake_artifacts())
        monkeypatch.setattr(
            main,
            "explain_payload",
            lambda payload, artifacts: {
                "fraud_probability": 0.91,
                "fraud_label": 1,
                "action": "BLOCK",
                "threshold": 0.5,
                "review_threshold": 0.35,
                "anomaly_flag": True,
                "anomaly_score": 0.12,
                "model_version": "xgb_final",
                "base_value": 0.05,
                "feature_count": 2,
                "top_features": [
                    {
                        "feature": "TransactionAmt",
                        "feature_value": 150.5,
                        "shap_value": 0.4,
                        "abs_shap_value": 0.4,
                    }
                ],
            },
        )

        response = await main.predict_and_explain(
            main.FraudPredictionRequest(TransactionAmt=150.5, TransactionDT=86450)
        )

        assert response.fraud_probability == 0.91
        assert response.fraud_label == 1
        assert response.action == "BLOCK"
        assert response.anomaly_score == 0.12
        assert response.base_value == 0.05
        assert response.feature_count == 2
        assert len(response.top_features) == 1
