"""Ultralytics → MLflow and W&B bridge callbacks.

Ultralytics has its own callback system: you register a function on an event
(e.g. on_fit_epoch_end) and it gets called by the trainer with the `trainer`
object. We use this to forward training metrics into our manually-managed
MLflow and W&B runs.

We deliberately do NOT use Ultralytics' built-in MLflow/W&B integrations:
  1. They create their own runs that fight with ours.
  2. They can't log the Hydra config as params.
  3. They can't register the model in MLflow Model Registry.

By bridging through our own callbacks, we keep the run lifecycle in our hands
and both trackers stay in sync (same step index, same metric names).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlflow

from football_tracker.training.utils import sanitize_metric_key
from football_tracker.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# MLflow callbacks
# ---------------------------------------------------------------------------


def on_fit_epoch_end(trainer: Any) -> None:
    """Forward per-epoch metrics to MLflow.

    `trainer.metrics` is a dict like:
        {
            "train/box_loss": 0.534,
            "train/cls_loss": 0.211,
            "metrics/precision(B)": 0.812,
            "metrics/mAP50(B)": 0.745,
            "metrics/mAP50-95(B)": 0.412,
        }
    `trainer.epoch` is the current 0-indexed epoch.
    """
    epoch = int(trainer.epoch)
    metrics = dict(trainer.metrics or {})
    if not metrics:
        return

    for key, value in metrics.items():
        try:
            clean_key = sanitize_metric_key(key)
            mlflow.log_metric(clean_key, float(value), step=epoch)
        except (ValueError, TypeError):
            continue


def on_train_end(trainer: Any) -> None:
    """Log Ultralytics artifacts (weights + plots) to MLflow at end of training."""
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        logger.warning("on_train_end: trainer has no save_dir, skipping artifact logging")
        return

    logger.info(f"Logging artifacts from {save_dir} to MLflow")

    weights_dir = save_dir / "weights"
    if weights_dir.exists():
        mlflow.log_artifacts(str(weights_dir), artifact_path="weights")

    for filename in [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "results.png",
        "results.csv",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]:
        f = save_dir / filename
        if f.exists():
            mlflow.log_artifact(str(f), artifact_path="plots")


def make_mlflow_callbacks() -> dict[str, Callable[[Any], None]]:
    """Return the MLflow callback map to register on a YOLO model."""
    return {
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


# ---------------------------------------------------------------------------
# W&B callbacks
# ---------------------------------------------------------------------------


def on_fit_epoch_end_wandb(trainer: Any) -> None:
    """Forward per-epoch metrics to W&B.

    Same data as the MLflow callback, same step index — both trackers stay
    in sync so you can compare runs across tools side by side.
    """
    try:
        import wandb as _wandb
    except ImportError:
        return
    if _wandb.run is None:  # type: ignore[attr-defined]
        return

    epoch = int(trainer.epoch)
    metrics = dict(trainer.metrics or {})
    if not metrics:
        return

    # W&B accepts arbitrary keys — no sanitization needed, but we keep the
    # same names as MLflow for consistency across dashboards.
    _wandb.log({"epoch": epoch, **metrics}, step=epoch)  # type: ignore[attr-defined]


def on_train_end_wandb(trainer: Any) -> None:
    """Log best.pt as a versioned W&B Model Artifact at end of training."""
    try:
        import wandb as _wandb
    except ImportError:
        return
    if _wandb.run is None:  # type: ignore[attr-defined]
        return

    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        return

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        logger.warning("on_train_end_wandb: best.pt not found, skipping W&B artifact")
        return

    artifact = _wandb.Artifact(  # type: ignore[attr-defined]
        name="football-yolov11n",
        type="model",
        description="YOLOv11n fine-tuned on football players detection dataset",
    )
    artifact.add_file(str(best_pt), name="best.pt")
    _wandb.log_artifact(artifact)  # type: ignore[attr-defined]
    logger.info("Logged best.pt as W&B model artifact")


def make_wandb_callbacks() -> dict[str, Callable[[Any], None]]:
    """Return the W&B callback map to register on a YOLO model."""
    return {
        "on_fit_epoch_end": on_fit_epoch_end_wandb,
        "on_train_end": on_train_end_wandb,
    }
