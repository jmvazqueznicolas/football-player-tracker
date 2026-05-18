.PHONY: help setup install install-dev lint format type-check test test-fast coverage clean download-data train tune serve api streamlit docker-build docker-run docker-push dvc-pull dvc-push mlflow-ui notebook gcp-setup

# Default target
.DEFAULT_GOAL := help

PROJECT_NAME := football-player-tracker
DOCKER_IMAGE := football-tracker
DOCKER_TAG := latest

# ----- Help -----
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ----- Setup -----
setup:  ## One-time project bootstrap: configure Poetry, pick Python 3.12, install everything
	@echo ">>> Checking Poetry version..."
	@poetry --version
	@echo ">>> Configuring Poetry to keep .venv inside the project..."
	poetry config virtualenvs.in-project true
	@echo ">>> Pointing Poetry at Python 3.12..."
	@command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found. Install it with: brew install python@3.12"; exit 1; }
	poetry env use python3.12
	@echo ">>> Installing dependencies and pre-commit hooks..."
	$(MAKE) install-dev
	@echo ">>> Done. Activate the venv with: poetry shell  (or use 'poetry run <cmd>')"

install:  ## Install runtime dependencies with Poetry
	poetry install --only main

install-dev:  ## Install all dependencies including dev
	poetry install
	poetry run pre-commit install

# ----- Quality -----
lint:  ## Run ruff and black checks
	poetry run ruff check src tests
	poetry run black --check src tests

format:  ## Auto-format code
	poetry run ruff check --fix src tests
	poetry run black src tests
	poetry run ruff format src tests

type-check:  ## Run mypy
	poetry run mypy src

# ----- Testing -----
test:  ## Run full test suite with coverage
	poetry run pytest

test-fast:  ## Run tests excluding slow ones
	poetry run pytest -m "not slow"

coverage:  ## Open coverage report in browser (after running test)
	poetry run pytest --cov-report=html
	@echo "Open htmlcov/index.html in your browser"

# ----- Data -----
download-data:  ## Download dataset from Roboflow (reads .env + configs/data/football.yaml)
	poetry run python scripts/download_dataset.py

# ----- Training and experimentation -----
train:  ## Train the baseline YOLOv11 model
	poetry run python -m football_tracker.training.train

tune:  ## Run Optuna hyperparameter search
	poetry run python -m football_tracker.training.tune

mlflow-ui:  ## Launch MLflow UI on port 5000
	poetry run mlflow ui --host 0.0.0.0 --port 5000

# ----- Serving -----
serve: api  ## Alias for api
api:  ## Run FastAPI server on port 8000
	poetry run uvicorn football_tracker.api.main:app --host 0.0.0.0 --port 8000 --reload

streamlit:  ## Run Streamlit demo on port 8501
	poetry run streamlit run src/football_tracker/api/streamlit_app.py

# ----- Docker -----
docker-build:  ## Build CPU Docker image
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-gpu:  ## Build GPU Docker image
	docker build -f docker/Dockerfile.gpu -t $(DOCKER_IMAGE):gpu .

docker-run:  ## Run Docker container locally
	docker run --rm -p 8000:8000 --env-file .env $(DOCKER_IMAGE):$(DOCKER_TAG)

# ----- Data and model versioning -----
dvc-pull:  ## Pull dataset and models from DVC remote
	poetry run dvc pull

dvc-push:  ## Push dataset and models to DVC remote
	poetry run dvc push

# ----- Notebooks -----
notebook:  ## Launch Jupyter Lab
	poetry run jupyter lab

# ----- Cleanup -----
clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ----- GCP -----
gcp-setup:  ## Print path to GCP setup guide
	@echo "Open docs/gcp_setup.md for step-by-step GCP setup"
