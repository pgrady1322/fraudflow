# ═══════════════════════════════════════════════════════════════════════
# FraudFlow — Developer Makefile
# ═══════════════════════════════════════════════════════════════════════

.PHONY: install dev lint format test serve tune explain drift pipeline docker clean help

PYTHON ?= python3
CONFIG ?= configs/pipeline.yaml

# ── Setup ───────────────────────────────────────────────────────────

install:  ## Install package
	$(PYTHON) -m pip install -e .

dev:  ## Install with dev + monitoring extras
	$(PYTHON) -m pip install -e ".[dev,monitoring]"
	pre-commit install

# ── Code Quality ────────────────────────────────────────────────────

lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format code with ruff
	ruff format src/ tests/

typecheck:  ## Run mypy type checking
	mypy src/ --ignore-missing-imports

# ── Testing ─────────────────────────────────────────────────────────

test:  ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html

# ── Pipeline ────────────────────────────────────────────────────────

pipeline:  ## Run full DVC pipeline
	dvc repro

train:  ## Train model
	$(PYTHON) -m src.training.train -c $(CONFIG)

evaluate:  ## Evaluate on test set
	$(PYTHON) -m src.training.evaluate -c $(CONFIG)

tune:  ## Run Optuna hyperparameter search
	$(PYTHON) -m src.training.tune -c $(CONFIG)

explain:  ## Generate SHAP explanations
	$(PYTHON) -m src.training.explain -c $(CONFIG)

drift:  ## Generate drift monitoring reports
	$(PYTHON) -m src.monitoring.drift -c $(CONFIG)

# ── Serving ─────────────────────────────────────────────────────────

serve:  ## Launch FastAPI serving endpoint
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:  ## Launch production serving (no reload)
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 4

# ── Docker ──────────────────────────────────────────────────────────

docker:  ## Build Docker image
	docker build -t fraudflow:latest .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 -v $(PWD)/models:/app/models fraudflow:latest

# ── MLflow ──────────────────────────────────────────────────────────

mlflow-ui:  ## Launch MLflow UI
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Cleanup ─────────────────────────────────────────────────────────

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean  ## Remove data, models, and MLflow runs too
	rm -rf data/processed/ data/splits/ models/registry/ mlruns/

# ── Help ────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
