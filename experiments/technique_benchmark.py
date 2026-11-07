"""Benchmark collaborative memory vs Mem0-style vs simple RAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.factory import backend_for_technique_label
from baselines.techniques import TECHNIQUE_LABELS, list_techniques, technique_label
from evaluation.ablation_metrics import AblationRunMetrics, extract_from_report
from evaluation.harness import BenchmarkReport, _detect_git_sha, _seed_everything
from experiments.aggregate import ExperimentSummary
from experiments.export import export_experiment
from experiments.runner import SCENARIO_FACTORIES
from experiments.technique_plots import generate_technique_plots
from baselines.runtime import configure_torch_safety, warn_if_risky_benchmark_env
from utils.config import load_config, reload_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_TECHNIQUES = list_techniques()
DEFAULT_SCENARIOS = list(SCENARIO_FACTORIES.keys())


class TechniqueBenchmarkRunner:
    """Run the same scenarios under each memory technique and export results."""

    def __init__(self, *, runs_dir: Path | str | None = None) -> None:
        self.runs_dir = Path(runs_dir or "experiments/runs")
        self._mem0_cache_dir: Path | None = None

    def run_sweep(
        self,
        *,
        techniques: list[str] | None = None,
        scenarios: list[str] | None = None,
        seed: int = 42,
    ) -> ExperimentSummary:
        technique_names = techniques or DEFAULT_TECHNIQUES
        scenario_names = scenarios or DEFAULT_SCENARIOS

        configure_torch_safety()
        for line in warn_if_risky_benchmark_env():
            logger.warning(line)

        reload_config()
        _seed_everything(seed)
        self._mem0_cache_dir = self.runs_dir / "techniques" / f"_mem0_cache_seed{seed}"
        summary = ExperimentSummary(
            seed=seed,
            baseline_variant="collaborative",
            git_sha=_detect_git_sha(),
        )

        for technique in technique_names:
            for scenario_name in scenario_names:
                if scenario_name not in SCENARIO_FACTORIES:
                    raise KeyError(
                        f"Unknown scenario {scenario_name!r}. "
                        f"Choose from: {sorted(SCENARIO_FACTORIES)}"
                    )
                logger.info("Technique cell: %s × %s", technique, scenario_name)
                cell = self._run_cell(scenario_name, technique=technique)
                summary.runs.append(cell)

        return summary

    def _run_cell(
        self,
        scenario_name: str,
        *,
        technique: str,
    ) -> AblationRunMetrics:
        if self._mem0_cache_dir is None:
            self._mem0_cache_dir = self.runs_dir / "techniques" / "_mem0_cache"
        factory = SCENARIO_FACTORIES[scenario_name]
        scenario = factory(technique=technique, mem0_cache_dir=self._mem0_cache_dir)
        try:
            scenario.setup()
            report = scenario.run()
        finally:
            scenario.teardown()

        if not isinstance(report, BenchmarkReport):
            raise TypeError(f"{scenario_name} must return BenchmarkReport")

        report = BenchmarkReport(
            scenario=report.scenario,
            metrics=report.metrics,
            expected=report.expected,
            queries=report.queries,
            notes={
                **report.notes,
                "technique": technique,
                "technique_label": technique_label(technique),
                "backend": (
                    scenario._backend.backend_info()
                    if getattr(scenario, "_backend", None) is not None
                    else {}
                ),
            },
        )
        metrics = extract_from_report(report, variant=technique)
        metrics.variant = technique
        return metrics

    def run_and_export(
        self,
        *,
        techniques: list[str] | None = None,
        scenarios: list[str] | None = None,
        seed: int = 42,
        run_id: str | None = None,
        generate_plots: bool = True,
    ) -> tuple[ExperimentSummary, dict[str, Path]]:
        cfg = reload_config()
        summary = self.run_sweep(techniques=techniques, scenarios=scenarios, seed=seed)
        tag = run_id or f"techniques_seed{seed}"
        out_dir = Path(cfg.experiments_runs_dir) / "techniques" / tag
        paths = export_experiment(summary, out_dir, run_id=tag)

        if "summary" in paths and paths["summary"].exists():
            payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
            payload["comparison_type"] = "memory_techniques"
            payload["technique_labels"] = {
                t: TECHNIQUE_LABELS.get(t, t) for t in (techniques or DEFAULT_TECHNIQUES)
            }
            payload["technique_implementations"] = {
                t: backend_for_technique_label(t) for t in (techniques or DEFAULT_TECHNIQUES)
            }
            payload["config_snapshot"] = cfg.public_summary()
            paths["summary"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if generate_plots:
            try:
                plot_paths = generate_technique_plots(summary, tag)
                paths.update(plot_paths)
            except ImportError as exc:
                logger.warning("Technique plots skipped: %s", exc)

        return summary, paths


def _print_summary_table(summary: ExperimentSummary) -> None:
    from experiments.aggregate import aggregate_by_variant

    grouped = aggregate_by_variant(summary)
    metrics = (
        "retrieval_precision",
        "retrieval_recall",
        "active_memory_accuracy",
        "long_horizon_consistency",
    )
    print("\n=== Technique comparison (mean across scenarios) ===")
    header = f"{'Technique':<28}" + "".join(f"{m:>18}" for m in metrics)
    print(header)
    print("-" * len(header))
    for technique in sorted(grouped.keys(), key=lambda t: (t != "collaborative", t)):
        label = technique_label(technique)
        agg = grouped[technique]
        vals = "".join(
            f"{float(agg.metrics.get(m, 0.0)):>18.3f}" for m in metrics
        )
        print(f"{label:<28}{vals}")


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(
        description="Benchmark mem0 vs simple RAG vs collaborative memory."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument(
        "--technique",
        action="append",
        help="technique name (repeatable): collaborative, simple_rag, mem0",
    )
    parser.add_argument("--scenario", action="append", help="scenario name (repeatable)")
    parser.add_argument("--list-techniques", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.list_techniques:
        for name in list_techniques():
            print(f"{name:16} → {technique_label(name)}")
        return 0
    if args.list_scenarios:
        for name in DEFAULT_SCENARIOS:
            print(name)
        return 0

    runner = TechniqueBenchmarkRunner(runs_dir=args.runs_dir)
    summary, paths = runner.run_and_export(
        techniques=args.technique,
        scenarios=args.scenario,
        seed=args.seed,
        generate_plots=not args.no_plots,
    )

    print(f"\nTechnique benchmark complete (seed={args.seed}, cells={len(summary.runs)})")
    for label, path in sorted(paths.items()):
        print(f"  {label}: {path}")

    _print_summary_table(summary)

    passed = sum(1 for r in summary.runs if r.passed)
    print(f"\nCells passing scenario thresholds: {passed}/{len(summary.runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
