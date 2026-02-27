"""Tests for src.training.models — model factory and helpers."""

import numpy as np
import pytest

from src.training.models import create_model, get_feature_columns


# ── create_model ────────────────────────────────────────────────────


class TestCreateModel:
    def test_xgboost_default(self):
        model = create_model("xgboost", {})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_xgboost_auto_scale(self):
        model = create_model("xgboost", {"scale_pos_weight": "auto"}, n_pos=100, n_neg=900)
        # scale_pos_weight ~ 9.0
        params = model.get_params()
        assert abs(params["scale_pos_weight"] - 9.0) < 0.1

    def test_random_forest(self):
        model = create_model("random_forest", {"n_estimators": 50})
        assert hasattr(model, "fit")
        params = model.get_params()
        assert params["n_estimators"] == 50

    def test_logistic_regression(self):
        model = create_model("logistic_regression", {"max_iter": 500})
        params = model.get_params()
        assert params["max_iter"] == 500

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("deep_neural_net", {})

    def test_xgboost_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)

        model = create_model("xgboost", {"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_random_forest_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)

        model = create_model("random_forest", {"n_estimators": 10})
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (100, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


# ── get_feature_columns ────────────────────────────────────────────


class TestGetFeatureColumns:
    def test_excludes_non_features(self):
        import pandas as pd

        df = pd.DataFrame({
            "txId": [1, 2],
            "timestep": [1, 1],
            "label": [0, 1],
            "feat_0": [0.1, 0.2],
            "feat_1": [0.3, 0.4],
        })
        cols = get_feature_columns(df)
        assert "txId" not in cols
        assert "timestep" not in cols
        assert "label" not in cols
        assert "feat_0" in cols
        assert "feat_1" in cols

    def test_empty_features(self):
        import pandas as pd

        df = pd.DataFrame({"txId": [1], "label": [0]})
        cols = get_feature_columns(df)
        assert len(cols) == 0
