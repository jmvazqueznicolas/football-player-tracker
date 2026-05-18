"""Dataset utilities: download from Roboflow, validate YOLO format, basic stats.

Implementation lands in Session 1 once the Roboflow dataset is chosen.
"""

from __future__ import annotations

from pathlib import Path


def validate_yolo_dataset(dataset_root: Path) -> dict:
    """Validate a YOLO-formatted dataset on disk.

    Expected layout:
        dataset_root/
            data.yaml
            images/{train,val,test}/*.jpg
            labels/{train,val,test}/*.txt

    Returns a dict with counts and any issues found.
    """
    raise NotImplementedError("Implement in Session 1.")


def class_distribution(dataset_root: Path, split: str = "train") -> dict[int, int]:
    """Count instances per class in a YOLO split."""
    raise NotImplementedError("Implement in Session 1.")
