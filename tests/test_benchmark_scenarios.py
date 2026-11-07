"""End-to-end tests for the deterministic benchmark scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.runner import default_harness
from benchmarks.scenarios import (
    CollaborativeRecallScenario,
    ContradictionResolutionScenario,
    LongHorizonScenario,
    TrustCalibrationScenario,
)
from evaluation.harness import BenchmarkReport, EvaluationHarness


SCENARIO_CLASSES = [
    ContradictionResolutionScenario,
    LongHorizonScenario,
    CollaborativeRecallScenario,
    TrustCalibrationScenario,
]


@pytest.mark.parametrize("scenario_cls", SCENARIO_CLASSES, ids=lambda c: c.name)
def test_scenario_meets_thresholds(scenario_cls) -> None:
    scenario = scenario_cls()
    scenario.setup()
    try:
        report = scenario.run()
    finally:
        scenario.teardown()

    assert isinstance(report, BenchmarkReport)
    assert report.passed, (
        f"{scenario.name} below thresholds: "
        f"metrics={report.metrics} expected={report.expected}"
    )


def test_harness_persists_manifest(tmp_path: Path) -> None:
    harness = EvaluationHarness(runs_dir=tmp_path)
    harness.register(ContradictionResolutionScenario())
    manifest = harness.run("contradiction_resolution_v1", seed=1234)

    out = tmp_path / "contradiction_resolution_v1_seed1234.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scenario"] == "contradiction_resolution_v1"
    assert payload["seed"] == 1234
    assert payload["passed"] is True
    assert payload["metrics"]["lineage_completeness"] == 1.0
    assert payload["queries"], "manifest should record per-query traces"
    assert manifest.passed


def test_default_harness_registers_all_scenarios(tmp_path: Path) -> None:
    harness = default_harness(runs_dir=tmp_path)
    expected = {
        "session_recall_v1",
        "contradiction_resolution_v1",
        "long_horizon_v1",
        "collaborative_recall_v1",
        "trust_calibration_v1",
    }
    assert expected.issubset(set(harness.list_scenarios()))


def test_run_all_produces_pass_for_each(tmp_path: Path) -> None:
    harness = default_harness(runs_dir=tmp_path)
    manifests = harness.run_all(seed=7)
    for name, manifest in manifests.items():
        assert manifest.passed, f"{name} failed: {manifest.metrics}"
