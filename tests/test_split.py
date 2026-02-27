"""Tests for src.data.split — temporal splitting."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.split import temporal_split


class TestTemporalSplit:
    @pytest.fixture
    def fake_dataset(self, tmp_path):
        """Create a minimal fake features.parquet and split paths."""
        rng = np.random.RandomState(42)
        n = 500

        # Create fake data across 10 timesteps
        df = pd.DataFrame({
            "txId": range(n),
            "timestep": rng.randint(1, 11, size=n),
            "label": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "feat_0": rng.randn(n),
            "feat_1": rng.randn(n),
            "feat_2": rng.randn(n),
        })

        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        features_path = processed_dir / "features.parquet"
        df.to_parquet(features_path, index=False)

        splits_dir = tmp_path / "data" / "splits"

        return features_path, splits_dir

    def test_split_creates_files(self, fake_dataset):
        features_path, splits_dir = fake_dataset
        stats = temporal_split(
            features_path, splits_dir,
            train_ts=(1, 7), val_ts=(8, 9), test_ts=(10, 10),
        )
        assert (splits_dir / "train.parquet").exists()
        assert (splits_dir / "val.parquet").exists()
        assert (splits_dir / "test.parquet").exists()
        assert (splits_dir / "split_stats.json").exists()

    def test_split_no_overlap(self, fake_dataset):
        features_path, splits_dir = fake_dataset
        temporal_split(
            features_path, splits_dir,
            train_ts=(1, 7), val_ts=(8, 9), test_ts=(10, 10),
        )

        train = pd.read_parquet(splits_dir / "train.parquet")
        val = pd.read_parquet(splits_dir / "val.parquet")
        test = pd.read_parquet(splits_dir / "test.parquet")

        train_ids = set(train["txId"])
        val_ids = set(val["txId"])
        test_ids = set(test["txId"])

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_split_temporal_ordering(self, fake_dataset):
        features_path, splits_dir = fake_dataset
        temporal_split(
            features_path, splits_dir,
            train_ts=(1, 7), val_ts=(8, 9), test_ts=(10, 10),
        )

        train = pd.read_parquet(splits_dir / "train.parquet")
        val = pd.read_parquet(splits_dir / "val.parquet")
        test = pd.read_parquet(splits_dir / "test.parquet")

        assert train["timestep"].max() <= 7
        assert val["timestep"].min() >= 8
        assert val["timestep"].max() <= 9
        assert test["timestep"].min() >= 10

    def test_split_stats_values(self, fake_dataset):
        features_path, splits_dir = fake_dataset
        stats = temporal_split(
            features_path, splits_dir,
            train_ts=(1, 7), val_ts=(8, 9), test_ts=(10, 10),
        )

        assert "train_samples" in stats
        assert "val_samples" in stats
        assert "test_samples" in stats
        assert stats["train_samples"] > 0
        assert stats["val_samples"] > 0
        assert stats["test_samples"] > 0

    def test_filters_unknown_labels(self, tmp_path):
        """Verify that label=-1 (unknown) samples are filtered out."""
        rng = np.random.RandomState(99)
        n = 200

        df = pd.DataFrame({
            "txId": range(n),
            "timestep": rng.randint(1, 11, size=n),
            "label": rng.choice([0, 1, -1], size=n, p=[0.5, 0.2, 0.3]),
            "feat_0": rng.randn(n),
        })

        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        features_path = processed_dir / "features.parquet"
        df.to_parquet(features_path, index=False)

        splits_dir = tmp_path / "data" / "splits"

        stats = temporal_split(
            features_path, splits_dir,
            train_ts=(1, 7), val_ts=(8, 9), test_ts=(10, 10),
        )
        total = stats["train_samples"] + stats["val_samples"] + stats["test_samples"]

        # Total should be less than n because label=-1 are filtered
        n_known = int((df["label"] != -1).sum())
        assert total <= n_known
