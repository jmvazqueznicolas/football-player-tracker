"""FastAPI service entry point. Implementation lands in Session 4."""

from __future__ import annotations

from fastapi import FastAPI

from football_tracker import __version__
from football_tracker.utils.logging import setup_logging

setup_logging()

app = FastAPI(
    title="Football Player Tracker",
    description="Detect and track football players, ball, referees and goalkeepers.",
    version=__version__,
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


@app.get("/", tags=["meta"])
def root() -> dict:
    """Service metadata."""
    return {
        "name": "football-player-tracker",
        "version": __version__,
        "docs": "/docs",
    }


# Endpoints /predict_image and /predict_video added in Session 4.
