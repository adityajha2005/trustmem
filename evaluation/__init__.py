"""Evaluation primitives and benchmark harness."""

from evaluation.harness import (
    BenchmarkReport,
    BenchmarkScenario,
    EvaluationHarness,
    RunManifest,
)
from evaluation.ablation_metrics import (
    CANONICAL_METRICS,
    AblationRunMetrics,
    extract_from_report,
)
from evaluation.metrics import (
    CollaborativeMetrics,
    active_memory_accuracy,
    context_efficiency,
    contradiction_resolution_rate,
    f1_at_k,
    long_horizon_consistency,
    mean,
    precision_at_k,
    ranking_inversion_rate,
    recall_at_k,
    stale_retrieval_rate,
    suppression_rate,
)

__all__ = [
    "AblationRunMetrics",
    "CANONICAL_METRICS",
    "BenchmarkReport",
    "BenchmarkScenario",
    "CollaborativeMetrics",
    "EvaluationHarness",
    "extract_from_report",
    "RunManifest",
    "active_memory_accuracy",
    "context_efficiency",
    "contradiction_resolution_rate",
    "f1_at_k",
    "long_horizon_consistency",
    "mean",
    "precision_at_k",
    "ranking_inversion_rate",
    "recall_at_k",
    "stale_retrieval_rate",
    "suppression_rate",
]
