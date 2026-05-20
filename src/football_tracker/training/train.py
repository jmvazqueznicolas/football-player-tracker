"""Main training entry point.

Trains a YOLO model on the football detection dataset, parameterized by Hydra
and instrumented with both MLflow and W&B for experiment tracking.

Both trackers run in parallel:
  - MLflow  → Model Registry, artifact storage, self-hosted (SQLite / GCS)
  - W&B     → richer dashboards, cross-run comparison, hosted service

Usage:
    poetry run python -m football_tracker.training.train
    poetry run python -m football_tracker.training.train training.epochs=3
    poetry run python -m football_tracker.training.train model=yolov11s training.epochs=50

For a smoke test on Mac (MPS, 2 epochs):
    poetry run python -m football_tracker.training.train \\
        training.epochs=2 training.batch=8 training.device=mps
"""

import os
from pathlib import Path

import hydra
import mlflow
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Disable Ultralytics' built-in MLflow and W&B integrations.
# We manage both run lifecycles ourselves so we can:
#   1. Log the full Hydra config as params.
#   2. Keep MLflow run ID in our hands for model registration.
#   3. Keep both trackers in sync (same step, same metric names).
os.environ["YOLO_MLFLOW"] = "false"
os.environ["YOLO_WANDB"] = "false"
try:
    from ultralytics.utils import SETTINGS as _ULTRA_SETTINGS

    _ULTRA_SETTINGS["mlflow"] = False
    _ULTRA_SETTINGS["wandb"] = False
except Exception:
    pass

from ultralytics import YOLO  # noqa: E402

from football_tracker.training.callbacks import (  # noqa: E402
    make_mlflow_callbacks,
    make_wandb_callbacks,
)
from football_tracker.training.utils import cfg_to_flat_params  # noqa: E402
from football_tracker.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger()


def _resolve_data_yaml(cfg: DictConfig) -> Path:
    """Resolve the dataset YAML path relative to the project root if needed."""
    yaml_path = Path(cfg.data.yaml_path)
    if not yaml_path.is_absolute():
        # Hydra runs from a fresh output dir, so relative paths must resolve
        # against the original project root, not the Hydra working dir.
        project_root = Path(hydra.utils.get_original_cwd())
        yaml_path = (project_root / yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    return yaml_path


def _materialize_absolute_data_yaml(original_yaml: Path, output_dir: Path) -> Path:
    """Write a resolved copy of the dataset YAML with an absolute `path:`.

    Ultralytics resolves the `path` field of a YOLO dataset YAML against its
    *global* SETTINGS["datasets_dir"] (typically ~/datasets), not against the
    location of the YAML itself. To avoid having to commit absolute paths
    into the repo, we rewrite the YAML at training time inside the Hydra
    output dir and point Ultralytics there.
    """
    spec = yaml.safe_load(original_yaml.read_text())
    # The directory that contains train/, valid/, test/ subfolders is the
    # original_yaml's parent. Make it absolute and inject it as `path:`.
    spec["path"] = str(original_yaml.parent.resolve())
    # Make sure the train/val/test entries stay simple relative names.
    for key in ("train", "val", "test"):
        if key in spec and isinstance(spec[key], str):
            # If user wrote "../train/images" or similar, normalize to "train/images"
            spec[key] = spec[key].lstrip("./").replace("..\\", "").replace("../", "")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = output_dir / "data_resolved.yaml"
    resolved_yaml.write_text(yaml.safe_dump(spec, sort_keys=False))
    return resolved_yaml


def _setup_mlflow(cfg: DictConfig) -> None:
    """Point MLflow at the configured tracking server and experiment."""
    uri = cfg.tracking_backend.mlflow.tracking_uri
    experiment = cfg.tracking_backend.mlflow.experiment_name
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    logger.info(f"MLflow tracking URI: {uri}")
    logger.info(f"MLflow experiment:   {experiment}")


def _setup_wandb(cfg: DictConfig, run_name: str) -> None:
    """Initialize a W&B run with the full Hydra config as its config object."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping W&B setup")
        return

    wcfg = cfg.tracking_backend.wandb
    entity = wcfg.entity if str(wcfg.entity) not in ("null", "None", "") else None

    wandb.init(  # type: ignore[attr-defined]
        project=wcfg.project,
        entity=entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(wcfg.tags),
        resume="allow",
    )
    logger.info(f"W&B run URL: {wandb.run.url}")  # type: ignore[attr-defined]


def _build_run_name(cfg: DictConfig) -> str:
    """Construct a human-readable MLflow run name."""
    return f"{cfg.model.name}_{cfg.training.epochs}ep_imgsz{cfg.training.imgsz}"


def _maybe_register_model(run_id: str, cfg: DictConfig) -> None:
    """Register best.pt in MLflow Model Registry with alias @candidate.

    MLflow 3.x requires artifacts to be logged via mlflow.log_model() to use
    the runs:/ shorthand in register_model(). Since we log weights with
    log_artifacts() (plain files), we use create_model_version() with the
    direct artifact URI instead — this bypasses the logged_model check and
    works correctly with our artifact layout.
    """
    if not cfg.tracking_backend.mlflow.enabled:
        return
    registry_name = cfg.model.registry_name
    try:
        client = mlflow.tracking.MlflowClient()
        artifact_uri = client.get_run(run_id).info.artifact_uri
        source = f"{artifact_uri}/weights/best.pt"
        mv = client.create_model_version(
            name=registry_name,
            source=source,
            run_id=run_id,
        )
        client.set_registered_model_alias(registry_name, "candidate", mv.version)
        logger.info(f"Registered '{registry_name}' v{mv.version} with alias @candidate")
    except Exception as e:
        logger.warning(f"Could not register model in MLflow Registry: {e}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a YOLO model with full MLflow instrumentation."""
    setup_logging()
    load_dotenv()

    logger.info("─" * 80)
    logger.info("Training configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logger.info("─" * 80)

    data_yaml = _resolve_data_yaml(cfg)
    logger.info(f"Original dataset YAML: {data_yaml}")

    if cfg.tracking_backend.mlflow.enabled:
        _setup_mlflow(cfg)

    run_name = _build_run_name(cfg)
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.tracking_backend.wandb.enabled:
        _setup_wandb(cfg, run_name)

    # Materialize a resolved data.yaml with absolute paths so Ultralytics
    # can find the dataset regardless of its global SETTINGS["datasets_dir"].
    resolved_data_yaml = _materialize_absolute_data_yaml(data_yaml, hydra_output_dir)
    logger.info(f"Resolved dataset YAML: {resolved_data_yaml}")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run name: {run_name}")
        logger.info(f"MLflow run ID:   {run_id}")

        # 1. Log the full Hydra config as MLflow params (flattened, dot notation)
        params = cfg_to_flat_params(cfg)
        # MLflow caps to 100 params per call; chunk if needed
        items = list(params.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i : i + 100]))
        logger.info(f"Logged {len(params)} hyperparameters to MLflow")

        # 2. Save the resolved config as a YAML artifact (very useful in retros)
        cfg_path = hydra_output_dir / "resolved_config.yaml"
        cfg_path.write_text(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(str(cfg_path), artifact_path="config")

        # 3. Tag the run with useful context
        mlflow.set_tags(
            {
                "model": cfg.model.name,
                "imgsz": str(cfg.training.imgsz),
                "epochs": str(cfg.training.epochs),
                "dataset": cfg.data.name,
                "seed": str(cfg.project.seed),
            }
        )

        # 4. Load model and attach callbacks for both trackers BEFORE training
        model = YOLO(cfg.model.weights)
        for event, fn in make_mlflow_callbacks().items():
            model.add_callback(event, fn)
        if cfg.tracking_backend.wandb.enabled:
            for event, fn in make_wandb_callbacks().items():
                model.add_callback(event, fn)
        logger.info(f"Loaded {cfg.model.weights} and attached MLflow + W&B callbacks")

        # 5. Train (using the resolved data YAML with absolute paths)
        training_kwargs = OmegaConf.to_container(cfg.training, resolve=True)
        results = model.train(
            data=str(resolved_data_yaml),
            project=str(hydra_output_dir),
            name="yolo_run",
            exist_ok=True,
            seed=cfg.project.seed,
            **training_kwargs,
        )

        # 6. Final summary metrics (single values, easy to read in both UIs)
        final_metrics: dict = {}
        if results is not None:
            box = getattr(results, "box", None)
            if box is not None:
                final_metrics = {
                    "final_mAP50": float(getattr(box, "map50", 0.0)),
                    "final_mAP50-95": float(getattr(box, "map", 0.0)),
                    "final_precision": float(getattr(box, "mp", 0.0)),
                    "final_recall": float(getattr(box, "mr", 0.0)),
                }
                mlflow.log_metrics(final_metrics)
                logger.info(f"Final metrics: {final_metrics}")

        # 7. Register the model in the MLflow Model Registry
        _maybe_register_model(run_id, cfg)

        # 8. Push final summary metrics to W&B and close the run
        if cfg.tracking_backend.wandb.enabled:
            try:
                import wandb

                if wandb.run is not None:  # type: ignore[attr-defined]
                    if final_metrics:
                        wandb.summary.update(final_metrics)  # type: ignore[attr-defined]
                    wandb.finish()  # type: ignore[attr-defined]
                    logger.info("W&B run finished")
            except ImportError:
                pass

        logger.info("─" * 80)
        logger.info(f"Done. MLflow run: {run_id}")
        logger.info(f"Output dir:       {hydra_output_dir}")
        logger.info("─" * 80)


if __name__ == "__main__":
    main()
