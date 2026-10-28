from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.train import _compute_scale_pos_weight, apply_smote, prepare_feature_matrix


class TestPrepareFeatureMatrix:
    def test_drops_target_and_id(self, merged_df):
        X = prepare_feature_matrix(
            merged_df,
            drop_cols=["TransactionID", "isFraud", "TransactionDT"],
        )
        assert "TransactionID" not in X.columns
        assert "isFraud" not in X.columns
        assert "TransactionDT" not in X.columns

    def test_no_object_columns(self, merged_df):
        X = prepare_feature_matrix(
            merged_df,
            drop_cols=["TransactionID", "isFraud", "TransactionDT"],
        )
        assert X.select_dtypes(include=["object", "category"]).shape[1] == 0


class TestScalePosWeight:
    def test_correct_ratio(self):
        y = np.array([0, 0, 0, 0, 1])  # 4 neg, 1 pos
        assert _compute_scale_pos_weight(y) == 4.0

    def test_no_positives_raises(self):
        y = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="No positive samples"):
            _compute_scale_pos_weight(y)


class TestApplySMOTE:
    def test_disabled_returns_original_data(self):
        X = pd.DataFrame({"f1": [0.1, 0.2, 0.3], "f2": [1.0, 2.0, 3.0]})
        y = np.array([0, 0, 1])

        X_out, y_out = apply_smote(X, y, smote_cfg={"enabled": False})

        assert X_out.equals(X)
        assert np.array_equal(y_out, y)

    def test_enabled_balances_classes_and_preserves_columns(self):
        X = pd.DataFrame({
            "f1": np.linspace(0.0, 1.0, 24),
            "f2": np.linspace(1.0, 2.0, 24),
        })
        y = np.array([0] * 20 + [1] * 4)

        X_out, y_out = apply_smote(
            X,
            y,
            smote_cfg={
                "enabled": True,
                "sampling_strategy": "auto",
                "k_neighbors": 5,
                "random_state": 42,
            },
        )

        counts = pd.Series(y_out).value_counts()
        assert counts[0] == counts[1] == 20
        assert list(X_out.columns) == list(X.columns)

    def test_raises_when_minority_class_has_single_sample(self):
        X = pd.DataFrame({"f1": [0.1, 0.2, 0.3], "f2": [1.0, 2.0, 3.0]})
        y = np.array([0, 0, 1])

        with pytest.raises(ValueError, match="minority class"):
            apply_smote(X, y, smote_cfg={"enabled": True, "k_neighbors": 5})
