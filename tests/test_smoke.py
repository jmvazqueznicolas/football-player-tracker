"""Smoke tests: package imports and basic structure are healthy."""

from __future__ import annotations


def test_package_imports() -> None:
    import football_tracker

    assert football_tracker.__version__


def test_subpackages_import() -> None:
    from football_tracker import (  # noqa: F401
        api,
        data,
        inference,
        models,
        tracking,
        training,
        utils,
    )


def test_logging_setup_runs() -> None:
    from football_tracker.utils.logging import get_logger, setup_logging

    setup_logging("DEBUG")
    logger = get_logger()
    logger.debug("smoke test")
