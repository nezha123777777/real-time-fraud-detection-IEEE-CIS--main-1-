"""
tests/test_evaluation.py
────────────────────────
Tests for src/utils/metrics.py and src.models/evaluate.py
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import compute_all_metrics, find_best_threshold


class TestComputeAllMetrics:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        m = compute_all_metrics(y_true, y_prob, threshold=0.5)
        assert m["roc_auc"] == 1.0
        assert m["recall"] == 1.0
        assert m["precision"] == 1.0
        assert m["f1"] == 1.0

    def test_random_classifier(self):
        rng = np.random.RandomState(0)
        y_true = rng.choice([0, 1], size=1000, p=[0.9, 0.1])
        y_prob = rng.uniform(0, 1, size=1000)
        m = compute_all_metrics(y_true, y_prob, threshold=0.5)
        # ROC-AUC should be around 0.5 for random
        assert 0.3 < m["roc_auc"] < 0.7

    def test_keys_present(self):
        m = compute_all_metrics(
            np.array([0, 1]), np.array([0.1, 0.9]), threshold=0.5
        )
        for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]:
            assert key in m


class TestFindBestThreshold:
    def test_finds_optimal(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        best_t, best_s = find_best_threshold(y_true, y_prob, metric="f1")
        # Threshold should be somewhere between 0.3 and 0.7
        assert 0.3 <= best_t <= 0.7
        assert best_s > 0.9  # should be near-perfect

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unsupported metric"):
            find_best_threshold(
                np.array([0, 1]), np.array([0.1, 0.9]), metric="accuracy"
            )
