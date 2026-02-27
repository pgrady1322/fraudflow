"""Tests for src.training.train — training pipeline helpers."""

import numpy as np

from src.training.train import compute_metrics, hybrid_resample


class TestHybridResample:
    def test_caps_majority(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10_000, 5)
        y = np.array([0] * 9_000 + [1] * 1_000)

        X_r, y_r = hybrid_resample(X, y, max_majority=5_000, target_minority_ratio=0.3, seed=42)
        n_maj = (y_r == 0).sum()
        assert n_maj <= 5_000

    def test_oversamples_minority(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1_000, 5)
        y = np.array([0] * 900 + [1] * 100)

        X_r, y_r = hybrid_resample(X, y, max_majority=900, target_minority_ratio=0.3, seed=42)
        n_min = (y_r == 1).sum()
        # target = 900 * 0.3 / 0.7 ≈ 386
        assert n_min > 100

    def test_preserves_shape(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 8)
        y = np.array([0] * 160 + [1] * 40)

        X_r, y_r = hybrid_resample(X, y, max_majority=200, target_minority_ratio=0.3, seed=42)
        assert X_r.shape[1] == 8
        assert len(X_r) == len(y_r)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])

        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"] == 1.0
        assert m["f1_illicit"] == 1.0
        assert m["auc_roc"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])

        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"] == 0.0
        assert m["f1_illicit"] == 0.0

    def test_metric_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.3, 0.7, 0.6, 0.4])

        m = compute_metrics(y_true, y_pred, y_prob)
        expected_keys = {
            "accuracy",
            "f1_illicit",
            "f1_macro",
            "precision_illicit",
            "recall_illicit",
            "auc_roc",
            "auc_pr",
        }
        assert expected_keys == set(m.keys())
