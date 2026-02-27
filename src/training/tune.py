"""
FraudFlow v0.1.0

tune.py — Optuna hyperparameter optimization with MLflow integration.

Usage: fraudflow tune -c configs/pipeline.yaml

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.training.models import create_model, get_feature_columns

logger = logging.getLogger("fraudflow.tune")


# ── Search spaces ───────────────────────────────────────────────────

SEARCH_SPACES: dict[str, dict] = {
    "xgboost": {
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 0.005, 0.3),
        "n_estimators": ("int", 100, 1500),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.3, 1.0),
        "min_child_weight": ("int", 1, 20),
        "reg_alpha": ("log_float", 1e-4, 10.0),
        "reg_lambda": ("log_float", 1e-4, 10.0),
        "gamma": ("log_float", 1e-4, 5.0),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int_or_none", 5, 30),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("categorical", ["sqrt", "log2", 0.5, 0.8]),
    },
    "logistic_regression": {
        "C": ("log_float", 1e-4, 100.0),
        "penalty": ("categorical", ["l1", "l2"]),
        "max_iter": ("int", 200, 2000),
    },
}


def _suggest_param(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    """Suggest a hyperparameter value from an Optuna trial."""
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    elif kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif kind == "log_float":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    elif kind == "int_or_none":
        use_none = trial.suggest_categorical(f"{name}_none", [True, False])
        if use_none:
            return None
        return trial.suggest_int(name, spec[1], spec[2])
    else:
        raise ValueError(f"Unknown param type: {kind}")


# ── Objective ───────────────────────────────────────────────────────

METRIC_FN = {
    "f1_illicit": lambda y, p: f1_score(y, (p > 0.5).astype(int), pos_label=1, zero_division=0),
    "auc_roc": lambda y, p: roc_auc_score(y, p),
    "auc_pr": lambda y, p: average_precision_score(y, p),
}


def make_objective(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    metric: str,
    n_folds: int,
    seed: int,
):
    """Create an Optuna objective function."""
    space = SEARCH_SPACES.get(model_type, {})
    score_fn = METRIC_FN.get(metric, METRIC_FN["f1_illicit"])

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest_param(trial, name, spec) for name, spec in space.items()}

        if model_type == "logistic_regression" and params.get("penalty") == "l1":
            params["solver"] = "saga"

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        model = create_model(model_type, params, n_pos, n_neg)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []

        for _fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            if model_type == "xgboost":
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            else:
                model.fit(X_tr, y_tr)

            y_prob = (
                model.predict_proba(X_vl)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_vl).astype(float)
            )
            scores.append(score_fn(y_vl, y_prob))

        mean_score = float(np.mean(scores))

        # Log to MLflow as a nested run
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric(f"cv_{metric}", mean_score)
            mlflow.log_metric("cv_std", float(np.std(scores)))

        return mean_score

    return objective


# ── Tuning pipeline ────────────────────────────────────────────────


def tune_pipeline(cfg: dict[str, Any]) -> dict:
    """
    Run Optuna hyperparameter search with MLflow tracking.

    Args:
        cfg: Full pipeline config.

    Returns:
        Dict with best params and score.
    """
    seed = cfg["project"].get("seed", 42)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    hp_cfg = train_cfg.get("hp_search", {})
    mlflow_cfg = cfg.get("mlflow", {})

    n_trials = hp_cfg.get("n_trials", 50)
    metric = hp_cfg.get("metric", "f1_illicit")
    n_folds = train_cfg.get("cross_validation", {}).get("n_folds", 5)

    # Setup MLflow
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    if not tracking_uri.startswith("http"):
        tracking_uri = str(Path(tracking_uri).resolve())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraudflow") + "_tuning")

    # Load data
    splits_dir = Path(cfg["data"]["splits_dir"])
    train_df = pd.read_parquet(splits_dir / "train.parquet")

    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols].fillna(0).values
    y = train_df["label"].values

    logger.info(f"Tuning {model_cfg['type']} | {n_trials} trials | metric={metric}")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"fraudflow_{model_cfg['type']}",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    with mlflow.start_run(run_name=f"tuning_{model_cfg['type']}"):
        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("metric", metric)
        mlflow.log_param("n_folds", n_folds)

        objective = make_objective(X, y, model_cfg["type"], metric, n_folds, seed)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Log best results
        best = study.best_trial
        mlflow.log_metric(f"best_{metric}", best.value)
        for k, v in best.params.items():
            mlflow.log_param(f"best_{k}", v)

    # Save results
    results_dir = Path("models/registry")
    results_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "best_params": best.params,
        f"best_{metric}": best.value,
        "n_trials": n_trials,
        "model_type": model_cfg["type"],
    }

    with open(results_dir / "tuning_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save optimization history as CSV (for plotting)
    history = []
    for trial in study.trials:
        row = {"trial": trial.number, "value": trial.value, "state": trial.state.name}
        row.update(trial.params)
        history.append(row)
    pd.DataFrame(history).to_csv(results_dir / "tuning_history.csv", index=False)

    logger.info(f"✓ Best {metric}: {best.value:.4f}")
    logger.info(f"  Best params: {best.params}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--config", "-c", required=True, help="Pipeline YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result = tune_pipeline(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
