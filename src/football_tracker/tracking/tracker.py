"""Wrapper around Ultralytics' built-in trackers (BoT-SORT, ByteTrack).

Implementation lands in Session 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class FootballTracker:
    """Run detection + tracking on a video clip, return persistent IDs per object."""

    def __init__(
        self,
        weights_path: str | Path,
        tracker_cfg: str | Path = "botsort.yaml",
        conf: float = 0.25,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.tracker_cfg = tracker_cfg
        self.conf = conf
        self._model: Any | None = None

    def track(self, video_path: str | Path, save: bool = True) -> dict:
        """Run tracking. Returns dict with trajectories indexed by track_id."""
        raise NotImplementedError("Implement in Session 4.")
