"""CLI runner for collaborative-memory benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.scenarios import (
    CollaborativeRecallScenario,
    ContradictionResolutionScenario,
    LongHorizonScenario,
    SessionRecallScenario,
    TrustCalibrationScenario,
)
from evaluation.harness import EvaluationHarness, RunManifest


def default_harness(runs_dir: Path | str | None = None) -> EvaluationHarness:
    """Build the default research benchmark suite."""
    harness = EvaluationHarness(runs_dir=runs_dir)
    for scenario in (
        SessionRecallScenario(),
        ContradictionResolutionScenario(),
        LongHorizonScenario(),
        CollaborativeRecallScenario(),
        TrustCalibrationScenario(),
    ):
        harness.register(scenario)
    return harness


def _format_manifest(manifest: RunManifest) -> str:
    status = "PASS" if manifest.passed else "FAIL"
    metric_lines = "  ".join(f"{k}={v:.3f}" for k, v in manifest.metrics.items())
    return f"[{status}] {manifest.scenario}  {metric_lines}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run collaborative memory benchmarks.")
    parser.add_argument("--scenario", default="all", help="scenario name or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--list", action="store_true", help="list available scenarios")
    args = parser.parse_args(argv)

    harness = default_harness(runs_dir=args.runs_dir)
    if args.list:
        for name in harness.list_scenarios():
            print(name)
        return 0

    if args.scenario == "all":
        manifests = harness.run_all(seed=args.seed)
        all_passed = True
        for manifest in manifests.values():
            print(_format_manifest(manifest))
            all_passed = all_passed and manifest.passed
        return 0 if all_passed else 1

    manifest = harness.run(args.scenario, seed=args.seed)
    print(_format_manifest(manifest))
    return 0 if manifest.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
