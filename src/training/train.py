"""
FraudFlow v0.1.0

train.py — Train model with MLflow experiment tracking.

DVC stage: train
Usage: python -m src.training.train --config configs/pipeline.yaml

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import resample

from src.training.models import create_model, get_feature_columns

logger = logging.getLogger("fraudflow.train")


# ── Resampling ──────────────────────────────────────────────────────


def hybrid_resample(
    X: np.ndarray,
    y: np.ndarray,
    max_majority: int = 50_000,
    target_minority_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hybrid resampling: cap majority class, oversample minority.

    Args:
        X: Feature matrix.
        y: Labels.
        max_majority: Max samples for majority class.
        target_minority_ratio: Target ratio of minority class.
        seed: Random seed.

    Returns:
        Resampled (X, y).
    """
    classes, counts = np.unique(y, return_counts=True)
    majority_cls = classes[np.argmax(counts)]
    minority_cls = classes[np.argmin(counts)]

    maj_mask = y == majority_cls
    min_mask = y == minority_cls

    X_maj, y_maj = X[maj_mask], y[maj_mask]
    X_min, y_min = X[min_mask], y[min_mask]

    # Undersample majority
    if len(X_maj) > max_majority:
        X_maj, y_maj = resample(
            X_maj, y_maj, n_samples=max_majority, random_state=seed, replace=False
        )

    # Oversample minority to target ratio
    target_min = int(len(X_maj) * target_minority_ratio / (1 - target_minority_ratio))
    target_min = max(target_min, len(X_min))

    if len(X_min) < target_min:
        X_min, y_min = resample(
            X_min, y_min, n_samples=target_min, random_state=seed, replace=True
        )

    X_out = np.vstack([X_maj, X_min])
    y_out = np.concatenate([y_maj, y_min])

    # Shuffle
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ── Training ────────────────────────────────────────────────────────


def train_pipeline(cfg: dict[str, Any]) -> dict:
    """
    Train a fraud detection model with MLflow tracking.

    Args:
        cfg: Full pipeline config dict.

    Returns:
        Dict of training metrics.
    """
    seed = cfg["project"].get("seed", 42)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    mlflow_cfg = cfg.get("mlflow", {})

    # ── Setup MLflow ────────────────────────────────────────────────
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    if not tracking_uri.startswith("http"):
        tracking_uri = str(Path(tracking_uri).resolve())
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mlflow_cfg.get("experiment_name", "fraudflow")
    mlflow.set_experiment(experiment_name)

    # ── Load data ───────────────────────────────────────────────────
    splits_dir = Path(cfg["data"]["splits_dir"])
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")

    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df["label"].values

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    logger.info(f"Train: {len(X_train):,} samples ({n_pos:,} illicit, {n_neg:,} licit)")
    logger.info(f"Val:   {len(X_val):,} samples")

    # ── Resampling ──────────────────────────────────────────────────
    resample_cfg = train_cfg.get("resampling", {})
    strategy = resample_cfg.get("strategy", "none")

    if strategy == "hybrid":
        X_train, y_train = hybrid_resample(
            X_train, y_train,
            max_majority=resample_cfg.get("max_majority", 50_000),
            target_minority_ratio=resample_cfg.get("target_minority_ratio", 0.3),
            seed=seed,
        )
        logger.info(f"After resampling: {len(X_train):,} samples")

    # ── Train with MLflow ───────────────────────────────────────────
    with mlflow.start_run(run_name=f"{model_cfg['type']}_{int(time.time())}") as run:
        # Log params
        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("resample_strategy", strategy)
        for k, v in model_cfg.get("params", {}).items():
            mlflow.log_param(f"model.{k}", v)

        # Create and fit model
        model = create_model(model_cfg["type"], model_cfg.get("params", {}), n_pos, n_neg)

        t0 = time.time()
        if model_cfg["type"] == "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Evaluate on validation set
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = compute_metrics(y_val, y_pred, y_prob)
        metrics["train_time_sec"] = round(train_time, 2)
        metrics["n_train_samples"] = len(X_train)
        metrics["n_val_samples"] = len(X_val)
        metrics["n_features"] = len(feature_cols)
        metrics["run_id"] = run.info.run_id

        # Log metrics to MLflow
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Register model if configured
        if mlflow_cfg.get("register_model", False):
            model_name = mlflow_cfg.get("model_name", "fraud-detector")
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {model_name}")

        logger.info(f"MLflow run: {run.info.run_id}")

    # ── Save locally (for DVC tracking) ─────────────────────────────
    registry_dir = Path("models/registry")
    registry_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = registry_dir / "model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    metrics_path = registry_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Confusion matrix CSV for DVC plots
    cm = confusion_matrix(y_val, y_pred)
    cm_labels = ["licit", "illicit"]
    cm_rows = []
    for i, actual in enumerate(cm_labels):
        for j, predicted in enumerate(cm_labels):
            cm_rows.append({"actual": actual, "predicted": predicted, "count": int(cm[i][j])})
    pd.DataFrame(cm_rows).to_csv(registry_dir / "confusion_matrix.csv", index=False)

    # Log feature importances if available
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(registry_dir / "feature_importance.csv", index=False)
        mlflow.log_artifact(str(registry_dir / "feature_importance.csv"))

    logger.info(f"✓ Model saved: {pkl_path}")
    logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
    logger.info(f"  F1 illicit: {metrics['f1_illicit']:.4f}")
    logger.info(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")

    return metrics


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute all evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_illicit": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_illicit": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_illicit": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--config", "-c", required=True, help="Pipeline YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    metrics = train_pipeline(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
