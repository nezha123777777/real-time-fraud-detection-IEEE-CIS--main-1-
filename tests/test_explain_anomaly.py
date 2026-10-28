from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.anomaly import score_anomalies, train_isolation_forest
from src.models.explain import explain_prediction


class TestExplainPrediction:
    def test_returns_ranked_top_features(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.random((40, 4)), columns=["f1", "f2", "f3", "f4"])
        y = np.array([0] * 30 + [1] * 10)

        model = xgb.XGBClassifier(
            n_estimators=15,
            max_depth=3,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X, y)

        result = explain_prediction(model, X.iloc[[0]], top_n=3)

        assert "base_value" in result
        assert result["feature_count"] == 4
        assert len(result["top_features"]) == 3
        assert (
            result["top_features"][0]["abs_shap_value"]
            >= result["top_features"][1]["abs_shap_value"]
        )


class TestIsolationForest:
    def test_train_and_score_isolation_forest(self):
        X = pd.DataFrame({
            "a": [0.1, 0.2, 0.15, 10.0, 0.18, 0.19],
            "b": [1.0, 1.1, 0.9, 15.0, 1.0, 1.2],
        })
        y = np.array([0, 0, 0, 1, 0, 0])

        model = train_isolation_forest(
            X,
            y,
            cfg={
                "enabled": True,
                "contamination": 0.2,
                "n_estimators": 50,
                "max_samples": "auto",
                "random_state": 42,
                "n_jobs": 1,
            },
        )

        assert model is not None
        scores, flags = score_anomalies(model, X)

        assert len(scores) == len(X)
        assert len(flags) == len(X)
        assert flags.dtype == bool
