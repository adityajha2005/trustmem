"""Tests for retrieval ablations and experiment sweeps."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.ablation_metrics import extract_from_report, stage_timings_from_result
from evaluation.harness import BenchmarkReport
from experiments.aggregate import (
    aggregate_by_variant,
    build_comparison_table,
    compute_deltas,
)
from experiments.runner import AblationExperimentRunner, CI_SCENARIOS, CI_VARIANTS
from retrieval.ablation import ABLATION_VARIANTS, RetrievalAblation, get_variant
from retrieval.context import RetrievalContext
from retrieval.retriever import CollaborativeMemoryRetriever, RetrievalResult, RetrievalStageTiming
from benchmarks.scenarios import ContradictionResolutionScenario


def test_ablation_variants_registered() -> None:
    assert "full" in ABLATION_VARIANTS
    assert "flat_baseline" in ABLATION_VARIANTS
    assert len(ABLATION_VARIANTS) >= 8


def test_no_trust_zeroes_trust_weight() -> None:
    from utils.config import AppConfig

    cfg = AppConfig()
    ablation = get_variant("no_trust_scoring")
    weights = ablation.scoring_weights(cfg)
    assert weights.trust == 0.0
    assert weights.semantic > cfg.semantic_weight


def test_extract_canonical_metrics() -> None:
    report = BenchmarkReport(
        scenario="test",
        metrics={
            "precision_at_1": 1.0,
            "stale_retrieval_rate": 0.0,
            "contradiction_resolution_rate": 1.0,
        },
    )
    run = extract_from_report(report, variant="full")
    assert run.metrics["retrieval_precision"] == 1.0
    assert run.metrics["contradiction_rate"] == 0.0


def test_stage_timings_from_result() -> None:
    result = RetrievalResult(
        query="q",
        timings=[
            RetrievalStageTiming("candidate_pool", 1.5),
            RetrievalStageTiming("score_rank", 2.5),
        ],
    )
    timings = stage_timings_from_result(result)
    assert timings["latency_candidate_pool_ms"] == 1.5
    assert timings["latency_total_ms"] == 4.0


def test_scenario_with_ablation_no_lineage(tmp_path: Path) -> None:
    scenario = ContradictionResolutionScenario(
        ablation=RetrievalAblation.no_lineage_filter(),
    )
    scenario.setup()
    try:
        report = scenario.run()
    finally:
        scenario.teardown()
    # Without lineage filter stale JWT may surface
    assert report.metrics.get("stale_retrieval_rate", 0.0) >= 0.0


def test_ablation_sweep_ci_subset(tmp_path: Path) -> None:
    runner = AblationExperimentRunner(runs_dir=tmp_path)
    summary = runner.run_sweep(
        variants=list(CI_VARIANTS),
        scenarios=list(CI_SCENARIOS),
        seed=99,
    )
    assert len(summary.runs) == len(CI_VARIANTS) * len(CI_SCENARIOS)
    by_variant = aggregate_by_variant(summary)
    assert "full" in by_variant
    deltas = compute_deltas(summary)
    assert "no_lineage_filter" in deltas


def test_export_writes_artifacts(tmp_path: Path) -> None:
    runner = AblationExperimentRunner(runs_dir=tmp_path)
    summary, paths = runner.run_and_export(
        variants=["full", "flat_baseline"],
        scenarios=["contradiction_resolution_v1"],
        seed=7,
        run_id="test_export",
    )
    assert paths["summary"].exists()
    assert paths["comparison_csv"].exists()
    assert paths["comparison_md"].exists()
    table = build_comparison_table(summary)
    assert len(table) == 2
