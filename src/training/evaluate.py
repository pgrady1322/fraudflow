"""
FraudFlow v0.1.0

evaluate.py — Evaluate trained model on held-out test set.

DVC stage: evaluate
Usage: python -m src.training.evaluate --config configs/pipeline.yaml

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.training.models import get_feature_columns

logger = logging.getLogger("fraudflow.evaluate")


def evaluate_pipeline(cfg: dict[str, Any]) -> dict:
    """
    Evaluate a trained model on the test set.

    Args:
        cfg: Full pipeline config dict.

    Returns:
        Dict of test metrics.
    """
    # ── Load model ──────────────────────────────────────────────────
    model_path = Path("models/registry/model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train stage first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── Load test data ──────────────────────────────────────────────
    splits_dir = Path(cfg["data"]["splits_dir"])
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["label"].values

    n_samples = len(X_test)
    n_pos = int((y_test == 1).sum())
    n_neg = int((y_test == 0).sum())
    logger.info(f"Test: {n_samples:,} samples ({n_pos:,} illicit, {n_neg:,} licit)")

    # ── Predict ─────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    # ── Metrics ─────────────────────────────────────────────────────
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_illicit": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_precision_illicit": float(
            precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "test_recall_illicit": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "test_auc_roc": float(roc_auc_score(y_test, y_prob)),
        "test_auc_pr": float(average_precision_score(y_test, y_prob)),
        "test_n_samples": n_samples,
        "test_n_illicit": n_pos,
        "test_n_licit": n_neg,
    }

    # ── Outputs ─────────────────────────────────────────────────────
    results_dir = Path("models/registry")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Test metrics JSON
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Classification report TXT
    report = classification_report(
        y_test, y_pred, target_names=["licit", "illicit"], zero_division=0
    )
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # ROC curve CSV (for DVC plots)
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    # Downsample for plotting if too many points
    if len(roc_df) > 500:
        idx = np.linspace(0, len(roc_df) - 1, 500, dtype=int)
        roc_df = roc_df.iloc[idx]
    roc_df.to_csv(results_dir / "roc_curve.csv", index=False)

    # Precision-Recall curve CSV (for DVC plots)
    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y_test, y_prob)
    pr_df = pd.DataFrame({"precision": precision_arr, "recall": recall_arr})
    if len(pr_df) > 500:
        idx = np.linspace(0, len(pr_df) - 1, 500, dtype=int)
        pr_df = pr_df.iloc[idx]
    pr_df.to_csv(results_dir / "precision_recall.csv", index=False)

    # Confusion matrix CSV
    cm = confusion_matrix(y_test, y_pred)
    cm_labels = ["licit", "illicit"]
    cm_rows = []
    for i, actual in enumerate(cm_labels):
        for j, predicted in enumerate(cm_labels):
            cm_rows.append({"actual": actual, "predicted": predicted, "count": int(cm[i][j])})
    pd.DataFrame(cm_rows).to_csv(results_dir / "test_confusion_matrix.csv", index=False)

    logger.info(f"✓ Test metrics saved: {metrics_path}")
    logger.info(f"  Accuracy:    {metrics['test_accuracy']:.4f}")
    logger.info(f"  F1 illicit:  {metrics['test_f1_illicit']:.4f}")
    logger.info(f"  AUC-ROC:     {metrics['test_auc_roc']:.4f}")
    logger.info(f"  AUC-PR:      {metrics['test_auc_pr']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model on test set")
    parser.add_argument("--config", "-c", required=True, help="Pipeline YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    metrics = evaluate_pipeline(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
