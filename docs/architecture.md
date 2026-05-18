# Architecture

> A living document. Updated at the end of each session.

## Components

| # | Component | Tool(s) | Where it runs |
|---|-----------|---------|---------------|
| 1 | Data versioning | DVC + GCS | Local + remote |
| 2 | Annotation review | Label Studio / FiftyOne | Local |
| 3 | Training | Ultralytics YOLOv11 | GCP Compute Engine (T4) |
| 4 | Experiment tracking | MLflow + Weights & Biases | Local MLflow + W&B Cloud |
| 5 | HPO | Optuna (TPE) | GCP Compute Engine |
| 6 | Model registry | MLflow Model Registry | Local with GCS artifacts |
| 7 | Error analysis | FiftyOne | Local notebook |
| 8 | Inference API | FastAPI + Uvicorn | Docker container |
| 9 | Demo frontend | Streamlit | Hugging Face Spaces |
| 10 | CI/CD | GitHub Actions | GitHub-hosted runners + GCP |
| 11 | Drift monitoring | Evidently AI + Streamlit | Same container as the API |

## Data flow

```
Roboflow Universe  ->  DVC remote (GCS)  ->  Local clone  ->  YOLO training
                                                 |
                                                 v
                                          MLflow + W&B logs
                                                 |
                                                 v
                                  MLflow Model Registry (Staging)
                                                 |
                                                 v
                              Gate: mAP50-95 >= threshold ?  --no-->  fail
                                                 |
                                                yes
                                                 v
                                  MLflow Model Registry (Production)
                                                 |
                                                 v
                                FastAPI service loads "Production" model
                                                 |
                                                 v
                                Streamlit demo on Hugging Face Spaces
                                                 |
                                                 v
                              Production logs  ->  Evidently drift report
```

## Reproducibility contract

Given the same Git SHA + DVC SHA + Hydra config, a training run reproduces within numerical noise. Concretely:

- Python and dependencies pinned in `poetry.lock`.
- Dataset pinned in `data/*.dvc` files.
- Hyperparameters in `configs/`, fully logged to MLflow + W&B.
- Random seeds set via `cfg.project.seed`.
- Hardware noted in MLflow tags (GPU model, CUDA version).
