"""Logging setup for research experiments and API runtime."""

from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_CONFIGURED = False


def setup_logging(level: LogLevel = "INFO", log_file: str | None = None) -> None:
    """Configure root logger once for console (and optional file) output."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    _CONFIGURED = True

    # TODO: Add structured JSON logging for experiment trace replay
    # TODO: Add per-agent log context (agent_id, session_id)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
