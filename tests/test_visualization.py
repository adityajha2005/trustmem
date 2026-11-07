"""Tests for experiment visualization and reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.ablation_metrics import AblationRunMetrics
from experiments.aggregate import ExperimentSummary
from experiments.export import export_experiment_with_artifacts
from experiments.load import load_summary_json, summary_from_dict
from experiments.reports import comparison_to_latex, generate_summary_report
from experiments.runner import AblationExperimentRunner, CI_SCENARIOS, CI_VARIANTS

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

from experiments.plots import generate_all_plots  # noqa: E402
from experiments.visualize import generate_figures_and_report  # noqa: E402


def _sample_summary() -> ExperimentSummary:
    runs = [
        AblationRunMetrics(
            variant="full",
            scenario="contradiction_resolution_v1",
            metrics={
                "retrieval_precision": 1.0,
                "retrieval_recall": 1.0,
                "contradiction_rate": 0.0,
                "stale_retrieval_rate": 0.0,
                "active_memory_accuracy": 1.0,
                "long_horizon_consistency": 0.0,
                "latency_candidate_pool_ms": 1.0,
                "latency_hybrid_search_ms": 2.0,
                "latency_total_ms": 5.0,
            },
        ),
        AblationRunMetrics(
            variant="no_lineage_filter",
            scenario="contradiction_resolution_v1",
            metrics={
                "retrieval_precision": 0.8,
                "retrieval_recall": 0.9,
                "contradiction_rate": 0.1,
                "stale_retrieval_rate": 0.2,
                "active_memory_accuracy": 0.85,
                "long_horizon_consistency": 0.0,
                "latency_candidate_pool_ms": 1.2,
                "latency_hybrid_search_ms": 2.1,
                "latency_total_ms": 5.5,
            },
        ),
        AblationRunMetrics(
            variant="full",
            scenario="long_horizon_v1",
            metrics={
                "retrieval_precision": 0.33,
                "retrieval_recall": 1.0,
                "long_horizon_consistency": 1.0,
                "stale_retrieval_rate": 0.0,
                "active_memory_accuracy": 1.0,
                "latency_total_ms": 8.0,
            },
        ),
    ]
    return ExperimentSummary(seed=7, baseline_variant="full", runs=runs, git_sha="abc123")


def test_load_summary_roundtrip(tmp_path: Path) -> None:
    summary = _sample_summary()
    from dataclasses import asdict

    payload = {
        "seed": summary.seed,
        "baseline_variant": summary.baseline_variant,
        "git_sha": summary.git_sha,
        "runs": [asdict(r) for r in summary.runs],
    }
    path = tmp_path / "test_summary.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_summary_json(path)
    assert loaded.seed == 7
    assert len(loaded.runs) == 3


def test_generate_all_plots(tmp_path: Path) -> None:
    summary = _sample_summary()
    plot_paths = generate_all_plots(summary, tmp_path / "plots")
    assert "ablation_comparison" in plot_paths
    png = tmp_path / "plots" / "ablation_comparison.png"
    pdf = tmp_path / "plots" / "ablation_comparison.pdf"
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists()


def test_summary_report_and_latex(tmp_path: Path) -> None:
    summary = _sample_summary()
    plots = generate_all_plots(summary, tmp_path / "plots")
    report = generate_summary_report(
        summary,
        tmp_path / "reports" / "run_report.md",
        plot_paths=plots,
        run_id="test_run",
    )
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Experiment report" in text
    from experiments.aggregate import build_comparison_table

    latex = comparison_to_latex(build_comparison_table(summary))
    assert r"\begin{tabular}" in latex


def test_export_with_artifacts_integration(tmp_path: Path) -> None:
    runner = AblationExperimentRunner(runs_dir=tmp_path)
    summary, paths = runner.run_and_export(
        variants=list(CI_VARIANTS),
        scenarios=list(CI_SCENARIOS),
        seed=11,
        run_id="viz_test",
        generate_plots=True,
    )
    assert paths["report"].exists()
    assert any(p.suffix == ".png" for p in paths.values() if hasattr(p, "suffix"))
