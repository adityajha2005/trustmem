"""Aggregate ablation sweep results into report-ready tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.ablation_metrics import CANONICAL_METRICS, STAGE_LATENCY_KEYS, AblationRunMetrics
from evaluation.metrics import mean


@dataclass
class ExperimentSummary:
    """Full ablation sweep output."""

    seed: int
    baseline_variant: str
    runs: list[AblationRunMetrics] = field(default_factory=list)
    git_sha: str | None = None

    def rows(self) -> list[dict[str, Any]]:
        return [r.to_row() for r in self.runs]

    def by_variant_scenario(self) -> dict[tuple[str, str], AblationRunMetrics]:
        return {(r.variant, r.scenario): r for r in self.runs}


@dataclass
class VariantAggregate:
    """Mean metrics for one variant across scenarios."""

    variant: str
    metrics: dict[str, float] = field(default_factory=dict)
    scenario_count: int = 0


def aggregate_by_variant(summary: ExperimentSummary) -> dict[str, VariantAggregate]:
    """Mean each canonical metric per variant across scenarios."""
    buckets: dict[str, list[AblationRunMetrics]] = {}
    for run in summary.runs:
        buckets.setdefault(run.variant, []).append(run)

    out: dict[str, VariantAggregate] = {}
    for variant, runs in buckets.items():
        keys = set()
        for run in runs:
            keys.update(run.metrics.keys())
        metrics: dict[str, float] = {}
        for key in sorted(keys):
            values = [r.metrics[key] for r in runs if key in r.metrics]
            if values:
                metrics[key] = mean(values)
        out[variant] = VariantAggregate(variant=variant, metrics=metrics, scenario_count=len(runs))
    return out


def compute_deltas(
    summary: ExperimentSummary,
    *,
    baseline: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Per-variant metric deltas vs baseline (variant - baseline).

    Negative delta on ``stale_retrieval_rate`` or ``ranking_inversion_rate`` is good.
    """
    base_name = baseline or summary.baseline_variant
    by_variant = aggregate_by_variant(summary)
    if base_name not in by_variant:
        return {}

    base_metrics = by_variant[base_name].metrics
    deltas: dict[str, dict[str, float]] = {}
    for variant, agg in by_variant.items():
        if variant == base_name:
            continue
        deltas[variant] = {
            key: agg.metrics.get(key, 0.0) - base_metrics.get(key, 0.0)
            for key in set(agg.metrics) | set(base_metrics)
        }
    return deltas


def build_comparison_table(
    summary: ExperimentSummary,
    *,
    metrics: tuple[str, ...] | None = None,
    baseline: str | None = None,
    include_deltas: bool = True,
) -> list[dict[str, Any]]:
    """
    Report-ready rows: one row per variant with mean metrics (+ optional Δ vs baseline).
    """
    metric_cols = metrics or CANONICAL_METRICS
    by_variant = aggregate_by_variant(summary)
    deltas = compute_deltas(summary, baseline=baseline) if include_deltas else {}
    base_name = baseline or summary.baseline_variant

    # Stable ordering: baseline first, then alphabetical.
    variants = sorted(by_variant.keys(), key=lambda v: (v != base_name, v))

    rows: list[dict[str, Any]] = []
    for variant in variants:
        agg = by_variant[variant]
        row: dict[str, Any] = {
            "variant": variant,
            "scenarios": agg.scenario_count,
        }
        for col in metric_cols:
            row[col] = round(agg.metrics.get(col, 0.0), 4)
        if include_deltas and variant != base_name and variant in deltas:
            for col in metric_cols:
                delta = deltas[variant].get(col, 0.0)
                row[f"delta_{col}"] = round(delta, 4)
        rows.append(row)
    return rows


def markdown_table(rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> str:
    """Render a list of dict rows as a GitHub-flavored markdown table."""
    if not rows:
        return "_No data._\n"
    cols = columns or list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def latency_summary_table(summary: ExperimentSummary) -> list[dict[str, Any]]:
    """Mean stage latencies per variant (ms)."""
    by_variant = aggregate_by_variant(summary)
    rows: list[dict[str, Any]] = []
    for variant in sorted(by_variant.keys()):
        agg = by_variant[variant]
        row: dict[str, Any] = {"variant": variant}
        for key in STAGE_LATENCY_KEYS:
            if key in agg.metrics:
                row[key] = round(agg.metrics[key], 3)
        rows.append(row)
    return rows
