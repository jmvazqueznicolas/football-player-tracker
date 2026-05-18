"""Inference wrapper around a YOLO model with consistent output schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Detection:
    """One detection in an image."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # absolute pixel coords


class Predictor:
    """Thin wrapper around an Ultralytics YOLO model.

    Implementation lands in Session 4.
    """

    def __init__(self, weights_path: str | Path, conf: float = 0.25, iou: float = 0.45) -> None:
        self.weights_path = Path(weights_path)
        self.conf = conf
        self.iou = iou
        self._model: Any | None = None

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights_path))

    def predict_image(self, image_path: str | Path) -> list[Detection]:
        raise NotImplementedError("Implement in Session 4.")

    def predict_video(self, video_path: str | Path, output_path: str | Path) -> dict:
        """Run detection + tracking, write annotated video, return trajectory JSON."""
        raise NotImplementedError("Implement in Session 4.")
