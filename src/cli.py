"""
FraudFlow v0.1.0

cli.py — Click CLI for the FraudFlow MLOps pipeline.

Entry point: fraudflow (defined in pyproject.toml)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import json
import logging
import sys

import click
import yaml


@click.group()
@click.version_option("0.1.0", prog_name="fraudflow")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def main(verbose: bool):
    """FraudFlow — MLOps pipeline for graph-based fraud detection."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── DVC Pipeline Stages ────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="Pipeline config YAML")
def download(config: str):
    """Stage 1: Download the Elliptic Bitcoin dataset."""
    from src.data.download import download_elliptic

    with open(config) as f:
        cfg = yaml.safe_load(f)

    download_elliptic(cfg)
    click.echo("✓ Download complete")


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="Pipeline config YAML")
def featurize(config: str):
    """Stage 2: Engineer graph topology features."""
    from src.features.engineer import engineer_features

    with open(config) as f:
        cfg = yaml.safe_load(f)

    stats = engineer_features(cfg)
    click.echo(f"✓ Features: {stats['n_features']} columns, {stats['n_samples']:,} rows")


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="Pipeline config YAML")
def split(config: str):
    """Stage 3: Create temporal train/val/test splits."""
    from src.data.split import temporal_split

    with open(config) as f:
        cfg = yaml.safe_load(f)

    stats = temporal_split(cfg)
    click.echo(
        f"✓ Split: train={stats['train_samples']:,}, "
        f"val={stats['val_samples']:,}, "
        f"test={stats['test_samples']:,}"
    )


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="Pipeline config YAML")
def train(config: str):
    """Stage 4: Train model with MLflow tracking."""
    from src.training.train import train_pipeline

    with open(config) as f:
        cfg = yaml.safe_load(f)

    metrics = train_pipeline(cfg)
    click.echo(f"✓ Train: F1={metrics['f1_illicit']:.4f}, AUC={metrics['auc_roc']:.4f}")


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="Pipeline config YAML")
def evaluate(config: str):
    """Stage 5: Evaluate model on held-out test set."""
    from src.training.evaluate import evaluate_pipeline

    with open(config) as f:
        cfg = yaml.safe_load(f)

    metrics = evaluate_pipeline(cfg)
    click.echo(
        f"✓ Test: F1={metrics['test_f1_illicit']:.4f}, "
        f"AUC-ROC={metrics['test_auc_roc']:.4f}, "
        f"AUC-PR={metrics['test_auc_pr']:.4f}"
    )


# ── Serve ───────────────────────────────────────────────────────────


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
@click.option("--model-path", default="models/registry/model.pkl", help="Path to model pickle")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, model_path: str, reload: bool):
    """Launch FastAPI model serving endpoint."""
    import uvicorn
    import src.serving.app as serving_app

    serving_app._model_path = model_path
    click.echo(f"Starting FraudFlow serving on {host}:{port}")
    uvicorn.run(
        "src.serving.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ── Pipeline ────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", required=True, type=click.Path(exists=True), help="Pipeline config YAML")
@click.option("--skip-download", is_flag=True, help="Skip data download")
def pipeline(config: str, skip_download: bool):
    """Run the full pipeline: download → featurize → split → train → evaluate."""
    from src.data.download import download_elliptic
    from src.data.split import temporal_split
    from src.features.engineer import engineer_features
    from src.training.train import train_pipeline
    from src.training.evaluate import evaluate_pipeline

    with open(config) as f:
        cfg = yaml.safe_load(f)

    if not skip_download:
        click.echo("── Stage 1: Download ──")
        download_elliptic(cfg)

    click.echo("── Stage 2: Featurize ──")
    engineer_features(cfg)

    click.echo("── Stage 3: Split ──")
    temporal_split(cfg)

    click.echo("── Stage 4: Train ──")
    train_metrics = train_pipeline(cfg)

    click.echo("── Stage 5: Evaluate ──")
    test_metrics = evaluate_pipeline(cfg)

    click.echo("\n═══ Pipeline Complete ═══")
    click.echo(f"  Train F1:  {train_metrics['f1_illicit']:.4f}")
    click.echo(f"  Test F1:   {test_metrics['test_f1_illicit']:.4f}")
    click.echo(f"  Test AUC:  {test_metrics['test_auc_roc']:.4f}")


if __name__ == "__main__":
    main()
