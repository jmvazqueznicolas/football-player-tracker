"""Helpers to load a YOLO model from local path or MLflow registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yolo(weights_path: str | Path) -> Any:
    """Load a YOLO model from a local .pt file."""
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def load_from_mlflow(model_name: str, stage: str = "Production") -> Any:
    """Load a model artifact from the MLflow Model Registry.

    Implementation lands in Session 2.
    """
    raise NotImplementedError("Implement in Session 2.")
