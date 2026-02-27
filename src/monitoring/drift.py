"""
FraudFlow v0.1.0

drift.py — Data and model drift monitoring with Evidently.

Generates HTML drift reports and JSON metrics for production monitoring.

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

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report

from src.training.models import get_feature_columns

logger = logging.getLogger("fraudflow.drift")


def build_column_mapping(feature_cols: list[str]) -> ColumnMapping:
    """Create Evidently column mapping for the Elliptic dataset."""
    return ColumnMapping(
        target="label",
        prediction="prediction",
        numerical_features=feature_cols,
    )


def data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> dict:
    """
    Generate a data drift report comparing reference and current data.

    Args:
        reference_df: Training/reference data.
        current_df: New/production data.
        feature_cols: Feature column names.
        output_dir: Where to save reports.

    Returns:
        Dict with drift summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    col_mapping = ColumnMapping(
        numerical_features=feature_cols,
    )

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    report.run(
        reference_data=reference_df[feature_cols],
        current_data=current_df[feature_cols],
        column_mapping=col_mapping,
    )

    # Save HTML report
    html_path = output_dir / "data_drift_report.html"
    report.save_html(str(html_path))
    logger.info(f"✓ Data drift report: {html_path}")

    # Extract JSON metrics
    result_json = report.as_dict()
    metrics_path = output_dir / "data_drift_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)

    # Parse summary
    n_drifted = 0
    n_total = len(feature_cols)
    for metric_result in result_json.get("metrics", []):
        result = metric_result.get("result", {})
        if "number_of_drifted_columns" in result:
            n_drifted = result["number_of_drifted_columns"]
            break

    summary = {
        "n_features": n_total,
        "n_drifted": n_drifted,
        "drift_ratio": round(n_drifted / max(n_total, 1), 3),
        "report_path": str(html_path),
    }

    logger.info(f"  Drifted features: {n_drifted}/{n_total} ({summary['drift_ratio']:.1%})")
    return summary


def model_performance_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
    output_dir: Path,
) -> dict:
    """
    Generate a model performance drift report.

    Compares model predictions on reference vs current data.

    Args:
        reference_df: Reference data with labels.
        current_df: Current data with labels.
        model: Trained model.
        feature_cols: Feature columns.
        output_dir: Output directory.

    Returns:
        Dict with performance summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add predictions to both datasets
    ref = reference_df.copy()
    cur = current_df.copy()

    for df, _name in [(ref, "reference"), (cur, "current")]:
        X = df[feature_cols].fillna(0).values
        df["prediction"] = model.predict(X)
        if hasattr(model, "predict_proba"):
            df["prediction_proba"] = model.predict_proba(X)[:, 1]

    col_mapping = build_column_mapping(feature_cols)

    report = Report(
        metrics=[
            ClassificationPreset(),
            TargetDriftPreset(),
        ]
    )

    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=col_mapping,
    )

    html_path = output_dir / "model_drift_report.html"
    report.save_html(str(html_path))
    logger.info(f"✓ Model drift report: {html_path}")

    result_json = report.as_dict()
    metrics_path = output_dir / "model_drift_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)

    summary = {"report_path": str(html_path)}
    return summary


def monitor(cfg: dict[str, Any]) -> dict:
    """
    Run full drift monitoring: data drift + model performance comparison.

    Compares training data (reference) with test data (current).

    Args:
        cfg: Pipeline config.

    Returns:
        Combined drift summary.
    """
    splits_dir = Path(cfg["data"]["splits_dir"])
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    feature_cols = get_feature_columns(train_df)

    output_dir = Path("models/registry/drift")

    # Data drift
    logger.info("── Data Drift Report ──")
    data_summary = data_drift_report(train_df, test_df, feature_cols, output_dir)

    # Model performance drift
    model_path = Path("models/registry/model.pkl")
    if model_path.exists():
        logger.info("── Model Drift Report ──")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_summary = model_performance_report(train_df, test_df, model, feature_cols, output_dir)
    else:
        logger.warning("No model found — skipping model drift report")
        model_summary = {}

    combined = {
        "data_drift": data_summary,
        "model_drift": model_summary,
    }

    with open(output_dir / "drift_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Data & model drift monitoring")
    parser.add_argument("--config", "-c", required=True, help="Pipeline YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result = monitor(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
