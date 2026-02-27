"""
FraudFlow v0.1.0

split.py — Temporal train/val/test split for the Elliptic dataset.

DVC stage: split
Usage: python -m src.data.split --config configs/pipeline.yaml

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("fraudflow.split")


def temporal_split(
    features_path: Path,
    splits_dir: Path,
    train_ts: tuple[int, int],
    val_ts: tuple[int, int],
    test_ts: tuple[int, int],
    seed: int = 42,
) -> dict:
    """
    Split features into train/val/test by Elliptic timestep.

    The Elliptic dataset has 49 timesteps of Bitcoin transactions.
    Temporal splits prevent data leakage from future → past.

    Args:
        features_path: Path to features.parquet
        splits_dir: Directory for output split parquets
        train_ts: (start, end) inclusive timestep range for training
        val_ts: (start, end) inclusive timestep range for validation
        test_ts: (start, end) inclusive timestep range for testing
        seed: Random seed

    Returns:
        Dict of split statistics.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    # Elliptic feature 0 (first column after txId) is the timestep
    ts_col = "timestep"
    if ts_col not in df.columns:
        raise KeyError(f"Column '{ts_col}' not found. Columns: {list(df.columns)[:10]}")

    # Filter to labelled rows only (label != "unknown")
    label_col = "label"
    if label_col in df.columns:
        known = df[df[label_col] != -1].copy()
        logger.info(f"Labelled rows: {len(known):,} / {len(df):,}")
    else:
        known = df.copy()

    # Split
    train = known[known[ts_col].between(*train_ts)]
    val = known[known[ts_col].between(*val_ts)]
    test = known[known[ts_col].between(*test_ts)]

    # Save
    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)

    stats = {
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "train_illicit_pct": float((train[label_col] == 1).mean()) if label_col in train else 0.0,
        "val_illicit_pct": float((val[label_col] == 1).mean()) if label_col in val else 0.0,
        "test_illicit_pct": float((test[label_col] == 1).mean()) if label_col in test else 0.0,
        "train_timesteps": list(train_ts),
        "val_timesteps": list(val_ts),
        "test_timesteps": list(test_ts),
    }

    stats_path = splits_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Split stats: {stats}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Temporal train/val/test split")
    parser.add_argument("--config", "-c", required=True, help="Pipeline YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    temporal_split(
        features_path=Path(data_cfg["processed_dir"]) / "features.parquet",
        splits_dir=Path(data_cfg["splits_dir"]),
        train_ts=tuple(data_cfg["train_timesteps"]),
        val_ts=tuple(data_cfg["val_timesteps"]),
        test_ts=tuple(data_cfg["test_timesteps"]),
        seed=cfg["project"].get("seed", 42),
    )


if __name__ == "__main__":
    main()
