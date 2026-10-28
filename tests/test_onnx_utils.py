from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.onnx_utils import load_onnx_session, predict_proba_onnx, save_model_onnx
from src.models.train import load_model, save_model


class TestOnnxModelExport:
    def test_onnx_predictions_match_xgboost(self, tmp_path):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.random((40, 3)), columns=["f1", "f2", "f3"])
        y = np.array([0] * 30 + [1] * 10)

        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X, y)

        onnx_path = save_model_onnx(
            model,
            feature_names=X.columns.tolist(),
            directory=tmp_path,
        )
        session = load_onnx_session(onnx_path)

        expected = model.predict_proba(X)[:, 1]
        actual = predict_proba_onnx(session, X)

        assert np.allclose(actual, expected, atol=1e-6)

    def test_native_xgboost_json_roundtrip(self, tmp_path):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.random((30, 2)), columns=["a", "b"])
        y = np.array([0] * 20 + [1] * 10)

        model = xgb.XGBClassifier(
            n_estimators=8,
            max_depth=2,
            eval_metric="logloss",
            random_state=0,
        )
        model.fit(X, y)

        save_model(model, directory=tmp_path)
        loaded = load_model(directory=tmp_path)

        expected = model.predict_proba(X)[:, 1]
        actual = loaded.predict_proba(X)[:, 1]

        assert np.allclose(actual, expected, atol=1e-6)
