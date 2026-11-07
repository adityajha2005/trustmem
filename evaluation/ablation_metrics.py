"""Canonical metric extraction for ablation experiment sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.harness import BenchmarkReport
from evaluation.metrics import mean
from retrieval.retriever import RetrievalResult


# Keys aggregated into results tables (order preserved for CSV/Markdown).
CANONICAL_METRICS: tuple[str, ...] = (
    "retrieval_precision",
    "retrieval_recall",
    "contradiction_rate",
    "stale_retrieval_rate",
    "active_memory_accuracy",
    "ranking_inversion_rate",
    "long_horizon_consistency",
    "context_efficiency",
    "lineage_completeness",
    "contradiction_resolution_rate",
    "trust_preservation_rate",
    "rerank_disruption_rate",
)

STAGE_LATENCY_KEYS: tuple[str, ...] = (
    "latency_candidate_pool_ms",
    "latency_scope_filter_ms",
    "latency_archive_filter_ms",
    "latency_lineage_filter_ms",
    "latency_hybrid_search_ms",
    "latency_flat_search_ms",
    "latency_score_rank_ms",
    "latency_rerank_ms",
    "latency_total_ms",
)


@dataclass
class AblationRunMetrics:
    """Normalized metrics for one (variant × scenario) cell."""

    variant: str
    scenario: str
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    query_count: int = 0

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"variant": self.variant, "scenario": self.scenario, "passed": self.passed}
        row.update(self.metrics)
        return row


def stage_timings_from_result(result: RetrievalResult) -> dict[str, float]:
    """Map retrieval stage timings to flat metric keys."""
    out: dict[str, float] = {}
    total = 0.0
    for timing in result.timings:
        key = f"latency_{timing.name}_ms"
        out[key] = timing.elapsed_ms
        total += timing.elapsed_ms
    out["latency_total_ms"] = total
    return out


def merge_stage_timings(traces: list[dict[str, Any]]) -> dict[str, float]:
    """Mean stage latency across query traces."""
    if not traces:
        return {}
    keys: set[str] = set()
    for trace in traces:
        keys.update(trace.get("stage_ms", {}).keys())
    merged: dict[str, float] = {}
    for key in keys:
        values = [
            float(trace["stage_ms"][key])
            for trace in traces
            if key in trace.get("stage_ms", {})
        ]
        if values:
            merged[f"latency_{key}_ms"] = mean(values)
    totals = [float(t.get("total_ms", 0.0)) for t in traces if "total_ms" in t]
    if totals:
        merged["latency_total_ms"] = mean(totals)
    return merged


def _first(report: BenchmarkReport, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in report.metrics:
            return float(report.metrics[key])
    return default


def extract_from_report(
    report: BenchmarkReport,
    *,
    variant: str,
) -> AblationRunMetrics:
    """
    Map scenario-specific metric names to canonical ablation columns.

    Scenarios use different internal names; this layer unifies them for sweeps.
    """
    m = report.metrics
    precision = _first(
        report,
        "precision_at_1",
        "mean_precision_at_3",
        "precision_at_1_post",
    )
    recall = _first(report, "mean_recall_at_3", "shared_recall_at_3", "recall_at_k")

    contradiction_rate = 0.0
    if "contradiction_resolution_rate" in m:
        contradiction_rate = 1.0 - float(m["contradiction_resolution_rate"])
    elif "contradiction_suppression_rate" in m:
        contradiction_rate = 1.0 - float(m["contradiction_suppression_rate"])

    stale = _first(report, "stale_retrieval_rate", "raw_stale_rate")
    active = _first(report, "active_memory_accuracy", "long_horizon_consistency")
    inversion = _first(report, "ranking_inversion_rate")
    horizon = _first(report, "long_horizon_consistency")
    context_eff = _first(report, "context_efficiency")
    lineage = _first(report, "lineage_completeness")
    resolution = _first(report, "contradiction_resolution_rate")

    canonical = {
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "contradiction_rate": contradiction_rate,
        "stale_retrieval_rate": stale,
        "active_memory_accuracy": active,
        "ranking_inversion_rate": inversion,
        "long_horizon_consistency": horizon,
        "context_efficiency": context_eff,
        "lineage_completeness": lineage,
        "contradiction_resolution_rate": resolution,
        "trust_preservation_rate": _first(
            report, "trust_preservation_rate", default=1.0
        ),
        "rerank_disruption_rate": _first(report, "rerank_disruption_rate"),
    }

    latency = merge_stage_timings(report.queries)
    canonical.update(latency)

    return AblationRunMetrics(
        variant=variant,
        scenario=report.scenario,
        metrics=canonical,
        passed=report.passed,
        query_count=len(report.queries),
    )


def attach_query_trace(
    trace: dict[str, Any],
    result: RetrievalResult,
) -> dict[str, Any]:
    """Enrich a query trace dict with retrieval stage latencies + rerank diag."""
    timings = stage_timings_from_result(result)
    trace["stage_ms"] = {
        k.removeprefix("latency_").removesuffix("_ms"): v for k, v in timings.items() if k != "latency_total_ms"
    }
    trace["total_ms"] = timings.get("latency_total_ms", 0.0)
    if result.rerank_diagnostics is not None:
        trace["rerank_diagnostics"] = result.rerank_diagnostics.to_dict()
    return trace
