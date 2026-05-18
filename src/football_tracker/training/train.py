"""Main training entry point.

Configurable via Hydra. Logs to MLflow and W&B in parallel.

Usage:
    python -m football_tracker.training.train
    python -m football_tracker.training.train model=yolov11s training.epochs=100
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a YOLO model with full experiment tracking.

    Implementation lands in Session 2:
      1. Initialize MLflow run and W&B run.
      2. Log Hydra config as a single artifact.
      3. Load Ultralytics YOLO with cfg.model.weights.
      4. Call model.train(**cfg.training) and stream metrics.
      5. Register best.pt in MLflow Model Registry under cfg.model.registry_name.
      6. Finish W&B run.
    """
    print(OmegaConf.to_yaml(cfg))
    raise NotImplementedError("Implement in Session 2.")


if __name__ == "__main__":
    main()
