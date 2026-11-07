"""Runtime guards for benchmark / Mem0 (macOS OpenMP, Python 3.14)."""

from __future__ import annotations

import os
import sys


def configure_torch_safety() -> None:
    """
    Reduce macOS ``Fatal Python error: Aborted`` from libomp when PyTorch loads.

    Call before importing ``torch``, ``sentence_transformers``, or ``mem0``.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Mem0 / PostHog background thread + OpenMP on pytest can abort on macOS.
    os.environ.setdefault("POSTHOG_DISABLED", "1")


def warn_if_risky_benchmark_env() -> list[str]:
    """Return human-readable warnings for the current process environment."""
    warnings: list[str] = []
    if sys.version_info >= (3, 14):
        warnings.append(
            "Python 3.14 detected: PyTorch/sentence-transformers are not fully "
            "stable. Prefer Python 3.12 or 3.13 for benchmarks."
        )
    offline = os.getenv("MEMORY_OFFLINE_RETRIEVAL", "").lower() in ("1", "true", "yes")
    if not offline:
        warnings.append(
            "MEMORY_OFFLINE_RETRIEVAL=0 loads BGE + cross-encoder + Mem0 (MiniLM) "
            "via PyTorch — high RAM and macOS OpenMP abort risk. For stable "
            "benchmarks use MEMORY_OFFLINE_RETRIEVAL=1."
        )
    return warnings
