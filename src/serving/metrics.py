"""
FraudFlow v0.1.0

metrics.py — Prometheus metrics instrumentation for the FastAPI serving endpoint.

Provides request counters, latency histograms, and prediction distribution gauges.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import time

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse

# ── Metrics ─────────────────────────────────────────────────────────

# Info metric
MODEL_INFO = Info("fraudflow_model", "Model metadata")

# Request metrics
REQUEST_COUNT = Counter(
    "fraudflow_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "fraudflow_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "fraudflow_predictions_total",
    "Total predictions made",
    ["predicted_class"],
)

PREDICTION_PROBABILITY = Histogram(
    "fraudflow_prediction_probability",
    "Distribution of P(illicit) scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PREDICTION_BATCH_SIZE = Histogram(
    "fraudflow_prediction_batch_size",
    "Number of samples per predict request",
    buckets=[1, 2, 5, 10, 25, 50, 100, 500, 1000],
)

INFERENCE_LATENCY = Histogram(
    "fraudflow_inference_latency_seconds",
    "Model inference time (excludes HTTP overhead)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
)

# Health gauge
MODEL_LOADED = Gauge(
    "fraudflow_model_loaded",
    "Whether the model is loaded (1=yes, 0=no)",
)


# ── Middleware ──────────────────────────────────────────────────────


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that records request counts and latencies."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        path = request.url.path

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(method=method, endpoint=path, status_code=response.status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)

        return response


# ── Metrics endpoint ────────────────────────────────────────────────


async def metrics_endpoint(request: Request) -> StarletteResponse:
    """Prometheus /metrics scrape endpoint."""
    return StarletteResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Helper functions for recording prediction metrics ───────────────


def record_predictions(predictions: list[int], probabilities: list[float]) -> None:
    """Record prediction metrics after an inference call."""
    PREDICTION_BATCH_SIZE.observe(len(predictions))

    for pred in predictions:
        PREDICTION_COUNT.labels(predicted_class=str(pred)).inc()

    for prob in probabilities:
        PREDICTION_PROBABILITY.observe(prob)


def record_inference_time(seconds: float) -> None:
    """Record raw model inference time."""
    INFERENCE_LATENCY.observe(seconds)


def update_model_status(loaded: bool, model_type: str = "") -> None:
    """Update model status gauge."""
    MODEL_LOADED.set(1 if loaded else 0)
    if model_type:
        MODEL_INFO.info({"model_type": model_type})
