"""
FraudFlow v0.1.0

app.py — FastAPI model serving endpoint for fraud detection.

Launch: uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import logging
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.metrics import (
    PrometheusMiddleware,
    metrics_endpoint,
    record_inference_time,
    record_predictions,
    update_model_status,
)

logger = logging.getLogger("fraudflow.serving")


# ── Globals ─────────────────────────────────────────────────────────

_model = None
_model_load_time: float = 0.0
_model_path: str = "models/registry/model.pkl"


# ── Schemas ─────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Input features for one or more transactions."""

    features: list[list[float]] = Field(
        ...,
        description="2D array of feature vectors, shape (n_samples, n_features)",
        min_length=1,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [
                        [0.1] * 166,  # Elliptic has 166 raw features
                    ]
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Fraud detection prediction results."""

    predictions: list[int] = Field(description="0=licit, 1=illicit")
    probabilities: list[float] = Field(description="P(illicit) for each sample")
    model_type: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Service health status."""

    status: str
    model_loaded: bool
    model_type: str | None
    uptime_sec: float


# ── Lifespan ────────────────────────────────────────────────────────

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model, _model_load_time, _start_time

    _start_time = time.time()

    model_path = Path(_model_path)
    if model_path.exists():
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        _model_load_time = time.time() - _start_time
        update_model_status(True, type(_model).__name__)
        logger.info(f"Model loaded from {model_path} in {_model_load_time:.2f}s")
    else:
        update_model_status(False)
        logger.warning(f"Model not found at {model_path}. /predict will return 503.")

    yield

    _model = None
    update_model_status(False)
    logger.info("Serving shutdown complete.")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudFlow",
    description="Fraud detection model serving with MLflow registry",
    version="0.1.0",
    lifespan=lifespan,
)

# Prometheus middleware and endpoint
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics_endpoint)


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_type=type(_model).__name__ if _model else None,
        uptime_sec=round(time.time() - _start_time, 2),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict fraud probability for transaction(s).

    Accepts a 2D array of feature vectors and returns predictions.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and deploy a model first.",
        )

    try:
        X = np.array(request.features, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid feature format: {e}") from None

    t0 = time.time()
    predictions = _model.predict(X).tolist()

    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba(X)[:, 1].tolist()
    else:
        probabilities = [float(p) for p in predictions]

    inference_sec = time.time() - t0
    inference_ms = inference_sec * 1000

    # Record Prometheus metrics
    record_inference_time(inference_sec)
    record_predictions(predictions, probabilities)

    return PredictionResponse(
        predictions=predictions,
        probabilities=probabilities,
        model_type=type(_model).__name__,
        inference_time_ms=round(inference_ms, 3),
    )


@app.get("/model/info")
async def model_info() -> dict[str, Any]:
    """Return model metadata."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    info: dict[str, Any] = {
        "model_type": type(_model).__name__,
        "model_load_time_sec": round(_model_load_time, 2),
    }

    # XGBoost / sklearn tree models expose n_features_in_
    if hasattr(_model, "n_features_in_"):
        info["n_features"] = int(_model.n_features_in_)

    if hasattr(_model, "get_params"):
        info["params"] = _model.get_params()

    return info
