# FraudFlow

**End-to-end MLOps pipeline for graph-based fraud detection** — DVC pipelines, MLflow experiment tracking, FastAPI model serving, and CI/CD.

Built on the [Elliptic Bitcoin Transaction Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) with graph topology features and temporal train/val/test splits to prevent data leakage.

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Download    │───▶│  Featurize  │───▶│    Split     │───▶│    Train    │───▶│  Evaluate   │
│  (Kaggle)   │    │ (Graph Topo)│    │  (Temporal)  │    │  (MLflow)   │    │  (Test Set) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       DVC stage 1       DVC stage 2       DVC stage 3       DVC stage 4       DVC stage 5
                                                                  │
                                                                  ▼
                                                          ┌─────────────┐
                                                          │  FastAPI    │
                                                          │  Serving    │
                                                          └─────────────┘
```

### Stack

| Component           | Technology         |
|---------------------|--------------------|
| Data versioning     | DVC 3.40+          |
| Experiment tracking | MLflow 2.10+       |
| HP optimization     | Optuna 3.5+ (TPE sampler) |
| Explainability      | SHAP 0.44+         |
| Drift monitoring    | Evidently 0.4+     |
| Model serving       | FastAPI + Uvicorn  |
| Serving metrics     | Prometheus (prometheus-client) |
| CI/CD               | GitHub Actions     |
| Code quality        | Ruff, mypy, pre-commit |
| Containerization    | Docker (multi-stage) |
| Models              | XGBoost, Random Forest, Logistic Regression |
| Graph features      | NetworkX (degree, clustering, PageRank, betweenness) |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/<your-username>/fraudflow.git
cd fraudflow
pip install -e ".[dev]"
```

### 2. Download Data

Requires a [Kaggle API token](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`.

```bash
fraudflow download -c configs/pipeline.yaml
```

### 3. Run the Full Pipeline

```bash
# Via CLI
fraudflow pipeline -c configs/pipeline.yaml

# Or via DVC
dvc repro
```

### 4. View Experiment Results

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

### 5. Serve the Model

```bash
fraudflow serve --port 8000

# Or via Docker
docker build -t fraudflow .
docker run -p 8000:8000 -v $(pwd)/models:/app/models fraudflow
```

### 6. Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, 0.2, ...]]}'
```

---

## Enhanced Features

### Hyperparameter Tuning (Optuna)

Bayesian optimization with TPE sampler, stratified K-fold CV, and MLflow nested run tracking:

```bash
fraudflow tune -c configs/pipeline.yaml
# or
make tune
```

Outputs `models/registry/tuning_results.json` and `models/registry/tuning_history.csv`.

### Model Explainability (SHAP)

Global and local feature importance using TreeExplainer (XGBoost/RF) or KernelExplainer (LR):

```bash
fraudflow explain -c configs/pipeline.yaml
# or
make explain
```

Produces bar, beeswarm, and waterfall plots plus `shap_importance.csv` in `models/registry/explanations/`.

### Data & Model Drift Monitoring (Evidently)

Detect feature distribution shifts and model performance degradation:

```bash
fraudflow drift -c configs/pipeline.yaml
# or
make drift
```

Generates HTML reports and JSON metrics in `models/registry/drift/`.

### Prometheus Serving Metrics

Live production metrics exposed at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `fraudflow_requests_total` | Counter | Total HTTP requests by method/endpoint/status |
| `fraudflow_predictions_total` | Counter | Predictions by class (illicit/licit) |
| `fraudflow_inference_latency_seconds` | Histogram | Model inference time |
| `fraudflow_model_loaded` | Gauge | 1 if model loaded, 0 otherwise |

### Makefile Targets

```bash
make help          # Show all targets
make install       # pip install -e .
make dev           # pip install -e ".[dev]"
make lint          # ruff check + mypy
make test          # pytest
make test-cov      # pytest + coverage report
make pipeline      # DVC repro
make tune          # Optuna HP search
make explain       # SHAP explanations
make drift         # Evidently drift reports
make serve         # FastAPI dev server
make docker        # Build Docker image
make clean         # Remove caches/artifacts
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

Runs ruff (lint + format), mypy, trailing whitespace, YAML/JSON/TOML validation, and large file checks on every commit.

---

## DVC Pipeline

The pipeline has 5 stages, each tracked by DVC for full reproducibility:

| Stage        | Script                    | Inputs               | Outputs                          |
|-------------|---------------------------|----------------------|----------------------------------|
| `download`  | `src/data/download.py`    | Kaggle API           | `data/raw/` CSV files            |
| `featurize` | `src/features/engineer.py`| Raw CSVs + edgelist  | `data/processed/features.parquet`|
| `split`     | `src/data/split.py`       | features.parquet     | `data/splits/{train,val,test}.parquet` |
| `train`     | `src/training/train.py`   | Train + val splits   | `models/registry/model.pkl`      |
| `evaluate`  | `src/training/evaluate.py`| Test split + model   | `models/registry/test_metrics.json` |

```bash
# Show the DAG
dvc dag

# Reproduce only changed stages
dvc repro

# Show metrics
dvc metrics show
```

---

## Project Structure

```
fraudflow/
├── configs/
│   └── pipeline.yaml          # Central configuration
├── src/
│   ├── cli.py                 # Click CLI entry point
│   ├── data/
│   │   ├── download.py        # Stage 1: Kaggle download
│   │   └── split.py           # Stage 3: Temporal split
│   ├── features/
│   │   └── engineer.py        # Stage 2: Graph feature engineering
│   ├── training/
│   │   ├── models.py          # Model factory (XGBoost, RF, LR)
│   │   ├── train.py           # Stage 4: MLflow-tracked training
│   │   ├── evaluate.py        # Stage 5: Test evaluation
│   │   ├── tune.py            # Optuna hyperparameter optimization
│   │   └── explain.py         # SHAP model explanations
│   ├── monitoring/
│   │   └── drift.py           # Evidently drift detection
│   └── serving/
│       ├── app.py             # FastAPI serving endpoint
│       └── metrics.py         # Prometheus instrumentation
├── tests/                     # 37 pytest tests
├── notebooks/
│   └── 01_pipeline_showcase.ipynb  # End-to-end demo notebook
├── .github/workflows/ci.yml   # CI: lint + test + Docker build
├── .pre-commit-config.yaml    # Code quality hooks
├── Makefile                   # 18 developer targets
├── Dockerfile                 # Multi-stage container
├── dvc.yaml                   # DVC pipeline definition
└── pyproject.toml             # Python project config
```

---

## API Endpoints

| Method | Path          | Description               |
|--------|---------------|---------------------------|
| GET    | `/health`     | Service health + model status |
| POST   | `/predict`    | Batch fraud predictions   |
| GET    | `/model/info` | Model metadata + params   |
| GET    | `/metrics`    | Prometheus metrics (text)  |

---

## Testing

```bash
pytest tests/ -v
```

37 tests covering:
- **Model factory** — creation, auto scale_pos_weight, fit/predict
- **Feature engineering** — graph construction, degree/clustering/PageRank
- **Temporal split** — file creation, no overlap, ordering, unknown filtering
- **Training helpers** — hybrid resampling, metric computation
- **FastAPI serving** — health, predict, batch, error cases, model info
- **Integration (E2E)** — full pipeline with synthetic data, serving round-trip

---

## Configuration

All pipeline parameters live in [`configs/pipeline.yaml`](configs/pipeline.yaml):

- **data**: paths, temporal split boundaries (timesteps 1–34 train, 35–42 val, 43–49 test)
- **features**: graph topology features, neighbor aggregation hops
- **model**: type (xgboost/random_forest/logistic_regression), hyperparameters
- **training**: resampling strategy, cross-validation folds
- **mlflow**: tracking URI, experiment name, model registry
- **serving**: host, port, model path

---

## License

MIT — see [LICENSE](LICENSE).
