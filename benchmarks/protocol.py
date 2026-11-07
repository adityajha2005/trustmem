"""Public protocol surface for collaborative memory benchmarks."""

from __future__ import annotations

from benchmarks._helpers import BaseScenario
from evaluation.harness import BenchmarkReport, BenchmarkScenario
from evaluation.metrics import CollaborativeMetrics

# Backward-compatible alias used by older scenarios returning CollaborativeMetrics.
CollaborativeScenario = BaseScenario

__all__ = [
    "BaseScenario",
    "BenchmarkReport",
    "BenchmarkScenario",
    "CollaborativeMetrics",
    "CollaborativeScenario",
]
