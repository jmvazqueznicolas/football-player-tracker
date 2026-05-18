"""Tests for the FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint() -> None:
    from football_tracker.api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint() -> None:
    from football_tracker.api.main import app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "football-player-tracker"
