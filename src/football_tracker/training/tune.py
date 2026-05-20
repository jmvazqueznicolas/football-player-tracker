"""Optuna Bayesian hyperparameter search.

Each trial trains the YOLO model for a reduced number of epochs (trial_epochs)
and returns the target metric. All trials are logged to MLflow as nested runs
under a single parent "HPO study" run so the full search is visible in one place.

The Optuna study is persisted to SQLite (optuna_study.db inside the Hydra
output dir) so it can be interrupted and resumed with --multirun or by simply
re-running the script — existing trials are skipped automatically.

Usage:
    # Default: 20 trials × 15 epochs each
    poetry run python -m football_tracker.training.tune

    # Override from CLI
    poetry run python -m football_tracker.training.tune tuning.n_trials=30
    poetry run python -m football_tracker.training.tune \\
        tuning.n_trials=10 tuning.trial_epochs=10 training.device=mps
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import hydra
import mlflow
import optuna
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Disable Ultralytics' built-in MLflow and W&B integrations — we own the lifecycle.
os.environ["YOLO_MLFLOW"] = "false"
os.environ["YOLO_WANDB"] = "false"
try:
    from ultralytics.utils import SETTINGS as _ULTRA_SETTINGS

    _ULTRA_SETTINGS["mlflow"] = False
    _ULTRA_SETTINGS["wandb"] = False
except Exception:
    pass

from ultralytics import YOLO  # noqa: E402

from football_tracker.training.callbacks import make_mlflow_callbacks  # noqa: E402
from football_tracker.training.train import (  # noqa: E402
    _materialize_absolute_data_yaml,
    _resolve_data_yaml,
    _setup_mlflow,
)
from football_tracker.training.utils import cfg_to_flat_params, sanitize_metric_key  # noqa: E402
from football_tracker.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRIC_ATTR_MAP: dict[str, str] = {
    "metrics/mAP50-95(B)": "map",
    "metrics/mAP50(B)": "map50",
    "metrics/precision(B)": "mp",
    "metrics/recall(B)": "mr",
}


def _sample_params(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Sample one set of hyperparameters from the search-space spec in optuna.yaml."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif kind == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search-space type '{kind}' for param '{name}'")
    return params


def _extract_metric(results: Any, metric_key: str) -> float:
    """Pull a scalar from an Ultralytics Results object by metric key."""
    if results is None:
        return 0.0
    box = getattr(results, "box", None)
    if box is None:
        return 0.0
    attr = _METRIC_ATTR_MAP.get(metric_key, "map")
    return float(getattr(box, attr, 0.0))


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------


def make_objective(
    cfg: DictConfig,
    resolved_data_yaml: Path,
    parent_run_id: str,
    hydra_output_dir: Path,
):
    """Return an Optuna objective closure bound to the current config."""

    def objective(trial: optuna.Trial) -> float:
        tuning_cfg = cfg.tuning
        search_space = OmegaConf.to_container(tuning_cfg.search_space, resolve=True)
        sampled = _sample_params(trial, search_space)  # type: ignore[arg-type]

        logger.info(f"─── Trial {trial.number:03d} ─── params: {sampled}")

        with mlflow.start_run(
            run_name=f"trial_{trial.number:03d}",
            nested=True,
            tags={
                "optuna_trial": str(trial.number),
                "parent_run_id": parent_run_id,
            },
        ) as child_run:
            child_run_id = child_run.info.run_id

            # Log sampled hyperparameters
            mlflow.log_params(sampled)

            # Build training kwargs: base from config, override with sampled values
            training_kwargs: dict[str, Any] = dict(
                OmegaConf.to_container(cfg.training, resolve=True)  # type: ignore[arg-type]
            )
            training_kwargs.update(sampled)
            training_kwargs["epochs"] = int(tuning_cfg.trial_epochs)

            # Load model and attach per-epoch MLflow callbacks
            model = YOLO(cfg.model.weights)
            for event, fn in make_mlflow_callbacks().items():
                model.add_callback(event, fn)

            trial_dir = hydra_output_dir / f"trial_{trial.number:03d}"

            try:
                results = model.train(
                    data=str(resolved_data_yaml),
                    project=str(trial_dir),
                    name="yolo_run",
                    exist_ok=True,
                    seed=cfg.project.seed,
                    **training_kwargs,
                )
            except Exception as exc:
                logger.error(f"Trial {trial.number} failed: {exc}")
                mlflow.set_tag("status", "FAILED")
                raise optuna.exceptions.TrialPruned() from exc

            metric_value = _extract_metric(results, tuning_cfg.metric)
            clean_key = sanitize_metric_key(tuning_cfg.metric)
            mlflow.log_metric(f"trial_{clean_key}", metric_value)
            mlflow.set_tag("status", "OK")

            logger.info(
                f"Trial {trial.number:03d} — {tuning_cfg.metric} = {metric_value:.4f}"
                f"  [run_id={child_run_id[:8]}]"
            )

        # Release GPU memory before the next trial
        del model
        gc.collect()

        return metric_value

    return objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Optuna HPO study with MLflow nested-run tracking."""
    setup_logging()
    load_dotenv()

    tuning_cfg = cfg.tuning

    logger.info("=" * 80)
    logger.info("HPO configuration:")
    logger.info("\n" + OmegaConf.to_yaml(tuning_cfg))
    logger.info("=" * 80)

    # Resolve dataset YAML (same logic as train.py)
    data_yaml = _resolve_data_yaml(cfg)
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    resolved_data_yaml = _materialize_absolute_data_yaml(data_yaml, hydra_output_dir)
    logger.info(f"Resolved dataset YAML: {resolved_data_yaml}")

    if cfg.tracking_backend.mlflow.enabled:
        _setup_mlflow(cfg)

    # ---------------------------------------------------------------------------
    # Create (or resume) Optuna study
    # ---------------------------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=cfg.project.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)

    # Store inside the Hydra output dir so every run has an isolated DB.
    # To resume a previous study, copy the .db file or pass storage explicitly.
    storage_path = hydra_output_dir / "optuna_study.db"
    storage_url = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=tuning_cfg.study_name,
        storage=storage_url,
        direction=tuning_cfg.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    n_trials = int(tuning_cfg.n_trials)
    trial_epochs = int(tuning_cfg.trial_epochs)
    logger.info(f"Study: '{tuning_cfg.study_name}' — {n_trials} trials × {trial_epochs} epochs")
    logger.info(f"Optuna storage: {storage_path}")

    # ---------------------------------------------------------------------------
    # MLflow parent run (wraps all trials as nested children)
    # ---------------------------------------------------------------------------
    study_run_name = f"hpo_{tuning_cfg.study_name}"

    with mlflow.start_run(run_name=study_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"MLflow parent run: {parent_run_id}")

        # Log study-level config as params
        flat_params = cfg_to_flat_params(
            OmegaConf.masked_copy(cfg, ["tuning", "model", "data", "project"])
        )
        items = list(flat_params.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i : i + 100]))

        mlflow.set_tags(
            {
                "hpo": "optuna",
                "sampler": tuning_cfg.sampler,
                "pruner": tuning_cfg.pruner,
                "n_trials": str(n_trials),
                "trial_epochs": str(trial_epochs),
                "metric": tuning_cfg.metric,
                "model": cfg.model.name,
                "dataset": cfg.data.name,
            }
        )

        # Run the search
        objective = make_objective(cfg, resolved_data_yaml, parent_run_id, hydra_output_dir)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=float(tuning_cfg.timeout_seconds) if tuning_cfg.timeout_seconds else None,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        # ---------------------------------------------------------------------------
        # Log best-trial results to parent run
        # ---------------------------------------------------------------------------
        best = study.best_trial
        best_metric_key = sanitize_metric_key(tuning_cfg.metric)
        mlflow.log_metric(f"best_{best_metric_key}", best.value)
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.set_tag("best_trial", str(best.number))

        # Persist the Optuna study DB as an MLflow artifact
        mlflow.log_artifact(str(storage_path), artifact_path="optuna")

        logger.info("=" * 80)
        logger.info(f"Best trial:  #{best.number}")
        logger.info(f"Best value:  {best.value:.4f}  ({tuning_cfg.metric})")
        logger.info(f"Best params: {best.params}")
        logger.info("=" * 80)
        logger.info(f"MLflow parent run: {parent_run_id}")
        logger.info(f"Output dir:        {hydra_output_dir}")
        logger.info("=" * 80)

    logger.info("HPO complete.")


if __name__ == "__main__":
    main()
