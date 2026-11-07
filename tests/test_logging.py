"""Logging utility tests."""

from __future__ import annotations

from utils.logging import get_logger


def test_get_logger() -> None:
    logger = get_logger("tests.logging")
    assert logger.name == "tests.logging"
    assert logger.level == 0  # NOTSET; defers to root handler
