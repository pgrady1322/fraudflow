"""
Integration test — end-to-end pipeline with synthetic Elliptic-like data.

Runs download(mock) → featurize → split → train → evaluate → serve
without needing Kaggle credentials. Safe for CI.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.data.split import temporal_split
from src.features.engineer import build_graph, compute_graph_features
from src.training.models import create_model, get_feature_columns
from src.training.train import compute_metrics, hybrid_resample


# ── Synthetic data factory ──────────────────────────────────────────


def make_synthetic_elliptic(tmp_path: Path, n_nodes: int = 2000, seed: int = 42):
    """
    Generate a synthetic dataset mimicking Elliptic's structure.

    Returns:
        Tuple of (raw_dir, processed_dir, splits_dir).
    """
    rng = np.random.RandomState(seed)

    # Nodes: 2000 transactions across 10 timesteps
    n_features = 20
    tx_ids = np.arange(n_nodes)
    timesteps = rng.randint(1, 11, size=n_nodes)

    # Generate feature matrix
    features = rng.randn(n_nodes, n_features)

    # Labels: ~80% licit (0), ~10% illicit (1), ~10% unknown (-1)
    labels = rng.choice([0, 1, -1], size=n_nodes, p=[0.75, 0.15, 0.10])

    # Make illicit nodes have slightly shifted features (learnable signal)
    illicit_mask = labels == 1
    features[illicit_mask, 0] += 1.5  # Shift feature 0
    features[illicit_mask, 1] -= 1.0  # Shift feature 1

    # Build DataFrame
    feat_cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=feat_cols)
    df.insert(0, "txId", tx_ids)
    df.insert(1, "timestep", timesteps)
    df["label"] = labels

    # Generate edges (random graph)
    n_edges = n_nodes * 3
    src = rng.randint(0, n_nodes, size=n_edges)
    dst = rng.randint(0, n_nodes, size=n_edges)
    edges_df = pd.DataFrame({"txId1": src, "txId2": dst})
    edges_df = edges_df[edges_df["txId1"] != edges_df["txId2"]].drop_duplicates()

    # Save
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    splits_dir = tmp_path / "data" / "splits"

    # Save processed features
    df.to_parquet(processed_dir / "features.parquet", index=False)
    edges_df.to_csv(raw_dir / "edges.csv", index=False)

    return raw_dir, processed_dir, splits_dir, df, edges_df


# ── Integration test ────────────────────────────────────────────────


class TestEndToEndPipeline:
    """Full pipeline integration test with synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.raw_dir, self.processed_dir, self.splits_dir, self.df, self.edges_df = (
            make_synthetic_elliptic(tmp_path)
        )

    def test_full_pipeline(self):
        """Run the entire pipeline end-to-end."""
        # ── Stage 2: Featurize (graph features) ────────────────────
        G = build_graph(self.edges_df)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

        enriched = compute_graph_features(self.df.copy(), G, ["degree", "pagerank"])
        assert "graph_degree" in enriched.columns
        assert "graph_pagerank" in enriched.columns

        # Save enriched features
        enriched.to_parquet(self.processed_dir / "features.parquet", index=False)

        # ── Stage 3: Split ──────────────────────────────────────────
        stats = temporal_split(
            features_path=self.processed_dir / "features.parquet",
            splits_dir=self.splits_dir,
            train_ts=(1, 7),
            val_ts=(8, 9),
            test_ts=(10, 10),
        )

        assert stats["train_samples"] > 0
        assert stats["val_samples"] > 0
        assert stats["test_samples"] > 0

        train_df = pd.read_parquet(self.splits_dir / "train.parquet")
        val_df = pd.read_parquet(self.splits_dir / "val.parquet")
        test_df = pd.read_parquet(self.splits_dir / "test.parquet")

        # Verify no label=-1 leaked through
        assert (train_df["label"] != -1).all()
        assert (val_df["label"] != -1).all()
        assert (test_df["label"] != -1).all()

        # ── Stage 4: Train ──────────────────────────────────────────
        feature_cols = get_feature_columns(train_df)
        assert len(feature_cols) > 0

        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["label"].values
        X_val = val_df[feature_cols].fillna(0).values
        y_val = val_df["label"].values

        # Resample
        X_train_r, y_train_r = hybrid_resample(
            X_train, y_train, max_majority=500, target_minority_ratio=0.3
        )

        # Create and train model
        n_pos = int((y_train_r == 1).sum())
        n_neg = int((y_train_r == 0).sum())
        model = create_model(
            "xgboost",
            {"n_estimators": 20, "max_depth": 4, "learning_rate": 0.1},
            n_pos, n_neg,
        )
        model.fit(X_train_r, y_train_r, eval_set=[(X_val, y_val)], verbose=False)

        # Validate predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        assert len(y_pred) == len(y_val)

        metrics = compute_metrics(y_val, y_pred, y_prob)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc_roc"] <= 1

        # With the signal we injected, model should do better than random
        assert metrics["auc_roc"] > 0.55, f"AUC-ROC too low: {metrics['auc_roc']:.3f}"

        # ── Stage 5: Evaluate on test set ───────────────────────────
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df["label"].values

        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

        assert 0 <= test_metrics["accuracy"] <= 1

        # ── Save model (for serving test) ───────────────────────────
        model_dir = self.tmp_path / "models"
        model_dir.mkdir()
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Verify model can be loaded back
        with open(model_dir / "model.pkl", "rb") as f:
            loaded = pickle.load(f)
        reload_pred = loaded.predict(X_test[:5])
        np.testing.assert_array_equal(reload_pred, y_test_pred[:5])

    def test_serving_with_synthetic_model(self):
        """Test the FastAPI serving endpoint with a synthetic model."""
        from fastapi.testclient import TestClient
        import src.serving.app as serving_module

        # Train a small model
        feature_cols = get_feature_columns(self.df[self.df["label"] != -1])
        known = self.df[self.df["label"] != -1]
        X = known[feature_cols].fillna(0).values
        y = known["label"].values

        model = create_model("xgboost", {"n_estimators": 5, "max_depth": 3})
        model.fit(X, y)

        # Inject into serving module
        serving_module._model = model
        serving_module._start_time = 0.0
        serving_module._model_load_time = 0.01

        client = TestClient(serving_module.app)

        # Health check
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is True

        # Single prediction
        sample = X[0].tolist()
        resp = client.post("/predict", json={"features": [sample]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["predictions"][0] in [0, 1]
        assert 0 <= data["probabilities"][0] <= 1

        # Batch prediction
        batch = X[:10].tolist()
        resp = client.post("/predict", json={"features": batch})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 10

        # Cleanup
        serving_module._model = None
