"""
FraudFlow v0.1.0

models.py — Model factory with MLflow-compatible wrappers.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logger = logging.getLogger("fraudflow.models")


def create_model(model_type: str, params: dict[str, Any], n_pos: int = 1, n_neg: int = 1):
    """
    Factory function to create a model from config.

    Args:
        model_type: One of 'xgboost', 'logistic_regression', 'random_forest'.
        params: Hyperparameters from config.
        n_pos: Count of positive (illicit) class samples.
        n_neg: Count of negative (licit) class samples.

    Returns:
        sklearn-compatible estimator.
    """
    params = params.copy()

    # Auto-compute scale_pos_weight if requested
    if params.get("scale_pos_weight") == "auto" and n_pos > 0:
        params["scale_pos_weight"] = n_neg / n_pos
        logger.info(f"Auto scale_pos_weight: {params['scale_pos_weight']:.2f}")

    if model_type == "xgboost":
        valid_keys = {
            "max_depth",
            "learning_rate",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
            "eval_metric",
            "random_state",
            "tree_method",
            "device",
            "gamma",
        }
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return XGBClassifier(
            use_label_encoder=False,
            verbosity=0,
            **filtered,
        )

    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 12),
            min_samples_split=params.get("min_samples_split", 5),
            class_weight="balanced",
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )

    elif model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            class_weight="balanced",
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_feature_columns(df) -> list[str]:
    """Get feature columns from a DataFrame (excludes metadata columns)."""
    exclude = {"txId", "timestep", "label", "class"}
    return [c for c in df.columns if c not in exclude]
