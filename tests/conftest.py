"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory holding small test fixtures (sample image, sample video)."""
    return Path(__file__).resolve().parent / "fixtures"
