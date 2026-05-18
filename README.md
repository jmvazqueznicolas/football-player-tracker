# Football Player Tracker

[![CI](https://github.com/USERNAME/football-player-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/football-player-tracker/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end Computer Vision + MLOps project that detects and tracks football players, the ball, referees and goalkeepers in match video, with full reproducibility, experiment tracking, automated CI/CD and drift monitoring in production.

> Live demo: **[link will go here once deployed]**
>
> Demo video: **[YouTube link will go here]**

---

## TL;DR

| Item | Detail |
|------|--------|
| **Problem** | Multi-class object detection + multi-object tracking on football match footage |
| **Model** | YOLOv11 (Ultralytics) fine-tuned on a custom football dataset, ONNX-exported for inference |
| **Tracker** | BoT-SORT (built-in Ultralytics) for persistent player IDs across frames |
| **Best mAP50** | _TBD after training_ |
| **Inference latency** | _TBD after benchmark (CPU and GPU)_ |
| **Stack** | PyTorch, Ultralytics, MLflow, Weights & Biases, DVC, Optuna, FiftyOne, FastAPI, Streamlit, Docker, GitHub Actions, Evidently, GCP |

---

## Table of contents

1. [Motivation](#motivation)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Quickstart](#quickstart)
5. [Dataset](#dataset)
6. [Training pipeline](#training-pipeline)
7. [Experiment tracking](#experiment-tracking)
8. [Hyperparameter optimization](#hyperparameter-optimization)
9. [Error analysis](#error-analysis)
10. [Serving and demo](#serving-and-demo)
11. [CI/CD](#cicd)
12. [Drift monitoring](#drift-monitoring)
13. [Tech decisions](#tech-decisions)
14. [Roadmap](#roadmap)

---

## Motivation

Standard "I trained a YOLO" tutorials skip the 80% of work that real ML engineering requires: dataset versioning, experiment tracking, hyperparameter search, error analysis, containerized serving, CI/CD with automated retraining, and drift monitoring in production. This project closes that gap end-to-end on a domain that is fun, visual and non-trivial: tracking individual players during live football footage.

## Architecture

_Architecture diagram will be added here in Session 8._

High level:

```
            +--------------------+
            |   Raw video / img  |
            +---------+----------+
                      |
                      v
            +---------+----------+
            |  Pre-processing    |
            +---------+----------+
                      |
                      v
            +---------+----------+         +---------------------+
            |  YOLOv11 detector  |-------->|  BoT-SORT tracker   |
            +---------+----------+         +----------+----------+
                      |                               |
                      v                               v
            +---------+-------------------------------+----------+
            |             Annotated video / JSON output         |
            +---------+-------------------------------+----------+
                      |
                      v
            +---------+----------+      +---------------------+
            |  FastAPI service   |<-----|  Streamlit frontend |
            +---------+----------+      +---------------------+
                      |
                      v
            +---------+----------+
            |  Drift monitoring  |
            +--------------------+
```

## Project structure

```
football-player-tracker/
├── configs/                  # Hydra configs (model, data, training, tracking)
├── data/                     # DVC-tracked datasets (not committed to Git)
├── docker/                   # Dockerfile (CPU + GPU)
├── docs/                     # Setup guides, architecture docs
├── models/                   # Trained model artifacts (DVC-tracked)
├── notebooks/                # EDA, error analysis (FiftyOne)
├── scripts/                  # One-off scripts (download data, convert formats)
├── src/football_tracker/     # Source code
│   ├── api/                  # FastAPI service
│   ├── data/                 # Dataset loading and validation
│   ├── inference/            # Inference and ONNX export
│   ├── models/               # Model loading and registry helpers
│   ├── tracking/             # BoT-SORT wrapper
│   ├── training/             # Train loop with MLflow + W&B
│   └── utils/                # Logging, metrics, helpers
├── tests/                    # pytest suite
├── .github/workflows/        # CI, training, deploy
├── pyproject.toml            # Poetry config
├── Makefile                  # Common commands
└── README.md
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/USERNAME/football-player-tracker.git
cd football-player-tracker

# 2. Install dependencies
poetry install
poetry shell

# 3. Pull dataset and models from DVC remote
dvc pull

# 4. Run tests
make test

# 5. Train a baseline model
make train

# 6. Serve API + Streamlit frontend
make serve
```

## Dataset

_To be completed in Session 1._

- **Source:** Roboflow Universe "football-players-detection" + custom frames extracted from public match footage.
- **Classes:** `player`, `ball`, `referee`, `goalkeeper`.
- **Annotation format:** YOLO (one `.txt` per image, normalized `cls x y w h`).
- **Versioning:** DVC with GCS remote.

## Training pipeline

_To be completed in Session 2._

## Experiment tracking

_To be completed in Session 2._

Both MLflow (self-hosted, GCS backend) and Weights & Biases (free tier) are used in parallel. MLflow handles model registry and stage promotion (Staging → Production). W&B provides richer dashboards for cross-run comparison.

## Hyperparameter optimization

_To be completed in Session 3._

Optuna Bayesian search over learning rate, momentum, weight decay and augmentation hyperparameters. Each trial logged to MLflow.

## Error analysis

_To be completed in Session 3._

FiftyOne is used to surface failure modes (small objects, occlusion, motion blur) and to slice metrics by image characteristics.

## Serving and demo

_To be completed in Session 5._

- FastAPI service exposing `/predict_image` and `/predict_video`.
- Streamlit frontend deployed on Hugging Face Spaces.
- Docker image (CPU multi-stage build) for portability.

## CI/CD

_To be completed in Session 6._

- **CI on PR:** lint + tests + Docker build.
- **Training pipeline:** triggered manually or by data change; promotes models to MLflow Production if metrics gate passes.
- **CD on merge to main:** rebuild and redeploy demo to Hugging Face Spaces.

## Drift monitoring

_To be completed in Session 7._

Production predictions are logged with image statistics. Evidently AI generates drift reports comparing production traffic to the training distribution. Alerts surface in a Streamlit dashboard.

## Tech decisions

_Living document — short rationales for each major choice._

- **Why YOLOv11?** State-of-the-art accuracy/speed tradeoff in 2025, native tracking support (BoT-SORT, ByteTrack), and excellent Ultralytics tooling.
- **Why both MLflow and W&B?** To demonstrate fluency with both standard tools. MLflow owns the Model Registry; W&B owns the visualization layer.
- **Why DVC over LakeFS?** DVC is lighter, Git-native and integrates with GCS in two commands.
- **Why Hugging Face Spaces for the demo?** Free Docker hosting with a public URL, no infra to manage.

## Roadmap

- [ ] Session 1 — Repo, dataset versioning, baseline training
- [ ] Session 2 — Training pipeline + MLflow + W&B
- [ ] Session 3 — Optuna HPO + FiftyOne error analysis
- [ ] Session 4 — ONNX export, tracking module, FastAPI
- [ ] Session 5 — Docker, Streamlit, public deploy
- [ ] Session 6 — CI/CD with GitHub Actions
- [ ] Session 7 — Drift monitoring with Evidently
- [ ] Session 8 — Polish, video, LinkedIn post

## License

MIT.

## Author

Jesús Manuel Vázquez Nicolás — built as a portfolio project for ML/MLOps Engineer roles in Computer Vision.
