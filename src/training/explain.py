"""
FraudFlow v0.1.0

explain.py — SHAP-based model explainability.

Generates global feature importance and per-prediction explanations.

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

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

from src.training.models import get_feature_columns

logger = logging.getLogger("fraudflow.explain")


def load_model_and_data(
    cfg: dict[str, Any],
) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    """Load trained model and test data."""
    model_path = Path("models/registry/model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    splits_dir = Path(cfg["data"]["splits_dir"])
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    feature_cols = get_feature_columns(test_df)
    X = test_df[feature_cols].fillna(0).values
    y = test_df["label"].values

    return model, X, y, feature_cols


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 1000,
    seed: int = 42,
) -> shap.Explanation:
    """
    Compute SHAP values for the model.

    Uses TreeExplainer for tree models, KernelExplainer as fallback.

    Args:
        model: Trained sklearn-compatible model.
        X: Feature matrix.
        feature_names: Column names.
        max_samples: Max samples for SHAP computation.
        seed: Random seed.
    """
    rng = np.random.RandomState(seed)

    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    model_name = type(model).__name__

    if model_name in ("XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier"):
        logger.info(f"Using TreeExplainer for {model_name}")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

        # For binary classifiers, TreeExplainer may return 3D array
        if isinstance(shap_values.values, np.ndarray) and shap_values.values.ndim == 3:
            # Take class 1 (illicit) SHAP values
            shap_values = shap.Explanation(
                values=shap_values.values[:, :, 1],
                base_values=shap_values.base_values[:, 1]
                if shap_values.base_values.ndim > 1
                else shap_values.base_values,
                data=shap_values.data,
                feature_names=feature_names,
            )
        else:
            shap_values.feature_names = feature_names
    else:
        logger.info(f"Using KernelExplainer for {model_name}")
        background = shap.kmeans(X_sample, min(50, len(X_sample)))
        predict_fn = (
            model.predict_proba
            if hasattr(model, "predict_proba")
            else model.predict
        )
        explainer = shap.KernelExplainer(lambda x: predict_fn(x)[:, 1], background)
        sv = explainer.shap_values(X_sample[:min(200, len(X_sample))])
        shap_values = shap.Explanation(
            values=sv,
            base_values=explainer.expected_value,
            data=X_sample[:min(200, len(X_sample))],
            feature_names=feature_names,
        )

    return shap_values


def generate_explanations(cfg: dict[str, Any]) -> dict:
    """
    Generate SHAP explanations and save plots + data.

    Args:
        cfg: Pipeline config.

    Returns:
        Dict with top features and summary stats.
    """
    seed = cfg["project"].get("seed", 42)
    model, X, y, feature_names = load_model_and_data(cfg)

    logger.info(f"Computing SHAP values for {len(X):,} test samples...")
    shap_values = compute_shap_values(model, X, feature_names, seed=seed)

    # Output directory
    output_dir = Path("models/registry/explanations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Global feature importance (bar plot) ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False, ax=ax)
    ax.set_title("Global Feature Importance (SHAP)")
    fig.tight_layout()
    fig.savefig(output_dir / "shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ Global importance plot saved")

    # ── Beeswarm (summary) plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(output_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ Beeswarm plot saved")

    # ── Mean absolute SHAP values table ─────────────────────────────
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(output_dir / "shap_importance.csv", index=False)

    # ── Per-class waterfall for top illicit and licit examples ──────
    if (y == 1).any() and (y == 0).any():
        # Find the highest-probability illicit sample
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X).astype(float)

        illicit_idx = np.where(y == 1)[0]
        top_illicit = illicit_idx[np.argmax(probs[illicit_idx])]

        licit_idx = np.where(y == 0)[0]
        top_licit = licit_idx[np.argmin(probs[licit_idx])]

        for label, idx in [("illicit", top_illicit), ("licit", top_licit)]:
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
            fig = plt.gcf()
            fig.suptitle(f"SHAP Waterfall — Typical {label.title()} Transaction", y=1.02)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"shap_waterfall_{label}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
            logger.info(f"✓ Waterfall plot ({label}) saved")

    # ── Summary stats ───────────────────────────────────────────────
    top_k = 10
    top_features = importance_df.head(top_k).to_dict("records")
    summary = {
        "n_samples_explained": len(shap_values.values),
        "n_features": len(feature_names),
        "top_features": top_features,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "explanation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ All explanations saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SHAP model explanations")
    parser.add_argument("--config", "-c", required=True, help="Pipeline YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    summary = generate_explanations(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
