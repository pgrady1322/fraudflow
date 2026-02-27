"""Tests for src.serving.app — FastAPI model serving."""

import numpy as np
import pickle
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

import src.serving.app as serving_module
from src.serving.app import app


@pytest.fixture
def trained_model(tmp_path):
    """Create a small trained model for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = (X[:, 0] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, model_path


@pytest.fixture
def client_with_model(trained_model):
    """TestClient with a loaded model."""
    model, model_path = trained_model
    serving_module._model = model
    serving_module._model_load_time = 0.01
    serving_module._start_time = 0.0

    client = TestClient(app)
    yield client

    serving_module._model = None


@pytest.fixture
def client_no_model():
    """TestClient without a model."""
    serving_module._model = None
    serving_module._start_time = 0.0
    client = TestClient(app)
    yield client


class TestHealth:
    def test_healthy_with_model(self, client_with_model):
        resp = client_with_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_type"] == "RandomForestClassifier"

    def test_degraded_without_model(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestPredict:
    def test_single_prediction(self, client_with_model):
        payload = {"features": [[0.5] * 10]}
        resp = client_with_model.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 1
        assert len(data["probabilities"]) == 1
        assert data["predictions"][0] in [0, 1]
        assert 0.0 <= data["probabilities"][0] <= 1.0

    def test_batch_prediction(self, client_with_model):
        rng = np.random.RandomState(42)
        features = rng.randn(5, 10).tolist()
        payload = {"features": features}
        resp = client_with_model.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 5
        assert len(data["probabilities"]) == 5

    def test_predict_no_model_503(self, client_no_model):
        payload = {"features": [[0.1] * 10]}
        resp = client_no_model.post("/predict", json=payload)
        assert resp.status_code == 503

    def test_predict_empty_features_422(self, client_with_model):
        payload = {"features": []}
        resp = client_with_model.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_inference_time_reported(self, client_with_model):
        payload = {"features": [[0.5] * 10]}
        resp = client_with_model.post("/predict", json=payload)
        data = resp.json()
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] >= 0


class TestModelInfo:
    def test_model_info(self, client_with_model):
        resp = client_with_model.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "RandomForestClassifier"
        assert "n_features" in data

    def test_model_info_no_model(self, client_no_model):
        resp = client_no_model.get("/model/info")
        assert resp.status_code == 503
