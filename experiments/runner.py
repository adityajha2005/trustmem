"""Ablation experiment runner: variant × scenario sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from benchmarks.scenarios import (
    CollaborativeRecallScenario,
    ContradictionResolutionScenario,
    LongHorizonScenario,
    TrustCalibrationScenario,
)
from evaluation.ablation_metrics import AblationRunMetrics, extract_from_report
from evaluation.harness import _detect_git_sha, _seed_everything
from experiments.aggregate import ExperimentSummary
from experiments.expected_failures import partition_failures
from experiments.export import export_experiment_with_artifacts
from retrieval.ablation import ABLATION_VARIANTS, RetrievalAblation, get_variant, list_variants
from utils.config import PROJECT_ROOT, load_config, reload_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_ABLATION_CONFIG = PROJECT_ROOT / "configs" / "ablation.yaml"

SCENARIO_FACTORIES: dict[str, type] = {
    "contradiction_resolution_v1": ContradictionResolutionScenario,
    "long_horizon_v1": LongHorizonScenario,
    "collaborative_recall_v1": CollaborativeRecallScenario,
    "trust_calibration_v1": TrustCalibrationScenario,
}

# CI-fast subset: one scenario × two variants
CI_VARIANTS = ("full", "no_lineage_filter")
CI_SCENARIOS = ("contradiction_resolution_v1",)


def _load_ablation_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


class AblationExperimentRunner:
    """
    Run reproducible component ablation sweeps over benchmark scenarios.

    Each cell is (variant, scenario) with isolated storage and deterministic seed.
    """

    def __init__(
        self,
        *,
        runs_dir: Path | str | None = None,
        baseline_variant: str = "full",
    ) -> None:
        self.runs_dir = Path(runs_dir or "experiments/runs")
        self.baseline_variant = baseline_variant

    def run_sweep(
        self,
        *,
        variants: list[str] | None = None,
        scenarios: list[str] | None = None,
        seed: int = 42,
    ) -> ExperimentSummary:
        variant_names = variants or list_variants()
        scenario_names = scenarios or list(SCENARIO_FACTORIES.keys())

        reload_config()
        _seed_everything(seed)
        summary = ExperimentSummary(
            seed=seed,
            baseline_variant=self.baseline_variant,
            git_sha=_detect_git_sha(),
        )

        for variant_name in variant_names:
            ablation = get_variant(variant_name)
            for scenario_name in scenario_names:
                if scenario_name not in SCENARIO_FACTORIES:
                    raise KeyError(
                        f"Unknown scenario {scenario_name!r}. "
                        f"Choose from: {sorted(SCENARIO_FACTORIES)}"
                    )
                logger.info("Ablation cell: variant=%s scenario=%s", variant_name, scenario_name)
                cell = self._run_cell(ablation, scenario_name)
                summary.runs.append(cell)

        return summary

    def _run_cell(self, ablation: RetrievalAblation, scenario_name: str) -> AblationRunMetrics:
        factory = SCENARIO_FACTORIES[scenario_name]
        scenario = factory(ablation=ablation)
        try:
            scenario.setup()
            report = scenario.run()
        finally:
            scenario.teardown()

        from evaluation.harness import BenchmarkReport

        if not isinstance(report, BenchmarkReport):
            raise TypeError(f"{scenario_name} must return BenchmarkReport for ablations")

        # Tag scenario name with variant for traceability in nested exports
        report = BenchmarkReport(
            scenario=report.scenario,
            metrics=report.metrics,
            expected=report.expected,
            queries=report.queries,
            notes={**report.notes, "ablation": ablation.to_dict()},
        )
        return extract_from_report(report, variant=ablation.name)

    def run_and_export(
        self,
        *,
        variants: list[str] | None = None,
        scenarios: list[str] | None = None,
        seed: int = 42,
        run_id: str | None = None,
        generate_plots: bool = True,
    ) -> tuple[ExperimentSummary, dict[str, Path]]:
        cfg = reload_config()
        summary = self.run_sweep(variants=variants, scenarios=scenarios, seed=seed)
        tag = run_id or f"ablation_seed{seed}"
        out_dir = Path(cfg.experiments_runs_dir) / "ablations" / tag
        paths = export_experiment_with_artifacts(
            summary,
            out_dir,
            run_id=tag,
            generate_plots=generate_plots,
        )
        # Attach reproducibility snapshot to summary JSON on disk
        if "summary" in paths and paths["summary"].exists():
            import json
            from dataclasses import asdict

            payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
            payload["config_snapshot"] = cfg.public_summary()
            paths["summary"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return summary, paths


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="WARNING")
    parser = argparse.ArgumentParser(description="Run collaborative-memory ablation sweeps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--config", type=Path, default=DEFAULT_ABLATION_CONFIG)
    parser.add_argument("--variant", action="append", help="variant name (repeatable)")
    parser.add_argument("--scenario", action="append", help="scenario name (repeatable)")
    parser.add_argument("--ci", action="store_true", help="fast CI subset (2 variants × 1 scenario)")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--baseline", default="full", help="baseline variant for deltas")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip matplotlib figure and report generation",
    )
    args = parser.parse_args(argv)

    if args.list_variants:
        for name in list_variants():
            print(name)
        return 0
    if args.list_scenarios:
        for name in sorted(SCENARIO_FACTORIES):
            print(name)
        return 0

    cfg = _load_ablation_config(args.config)
    variants = args.variant or cfg.get("variants")
    scenarios = args.scenario or cfg.get("ablation_scenarios")

    if args.ci:
        variants = list(CI_VARIANTS)
        scenarios = list(CI_SCENARIOS)

    runner = AblationExperimentRunner(
        runs_dir=args.runs_dir,
        baseline_variant=args.baseline,
    )
    summary, paths = runner.run_and_export(
        variants=variants,
        scenarios=scenarios,
        seed=args.seed,
        generate_plots=not args.no_plots,
    )

    print(f"Ablation sweep complete (seed={args.seed}, cells={len(summary.runs)})")
    for label, path in sorted(paths.items()):
        print(f"  {label}: {path}")

    failed = [r for r in summary.runs if not r.passed]
    if failed:
        expected_failed, unexpected_failed = partition_failures(failed)
        print(
            f"INFO: {len(failed)} cells below scenario thresholds "
            f"({len(expected_failed)} expected negative controls, "
            f"{len(unexpected_failed)} unexpected)"
        )
        if unexpected_failed:
            for run in unexpected_failed:
                print(f"  UNEXPECTED: {run.variant} × {run.scenario}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
