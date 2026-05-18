"""Optuna Bayesian hyperparameter search.

Each trial is logged to MLflow as a nested run.

Usage:
    python -m football_tracker.training.tune
    python -m football_tracker.training.tune tuning.n_trials=30
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Optuna HPO.

    Implementation lands in Session 3:
      - Create study with TPE sampler.
      - Define objective() that trains a short run with sampled hyperparameters.
      - Return mAP50-95 to maximize.
      - Log each trial to MLflow.
      - Persist Optuna study to optuna_storage.db so it can be resumed.
    """
    raise NotImplementedError("Implement in Session 3.")


if __name__ == "__main__":
    main()
