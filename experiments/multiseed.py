"""Multi-seed ablation sweeps for real-model variance reporting."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evaluation.metrics import mean
from experiments.aggregate import ExperimentSummary
from experiments.export import export_experiment_with_artifacts
from experiments.runner import AblationExperimentRunner, _load_ablation_config, DEFAULT_ABLATION_CONFIG
from evaluation.ablation_metrics import CANONICAL_METRICS
from utils.config import reload_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def run_multiseed(
    seeds: list[int],
    *,
    variants: list[str] | None = None,
    scenarios: list[str] | None = None,
    runs_dir: Path | str | None = None,
    generate_plots: bool = True,
) -> Path:
    """
    Run the ablation grid for each seed; write a combined manifest.

    Returns path to ``multiseed_summary.json``.
    """
    cfg = reload_config()
    runner = AblationExperimentRunner(runs_dir=runs_dir or cfg.experiments_runs_dir)
    all_summaries: list[ExperimentSummary] = []

    for seed in seeds:
        tag = f"ablation_seed{seed}"
        logger.info("Multi-seed run seed=%d", seed)
        summary, _paths = runner.run_and_export(
            variants=variants,
            scenarios=scenarios,
            seed=seed,
            run_id=tag,
            generate_plots=generate_plots,
        )
        all_summaries.append(summary)

    combined_dir = Path(cfg.experiments_runs_dir) / "ablations" / "multiseed"
    combined_dir.mkdir(parents=True, exist_ok=True)
    out_path = combined_dir / f"multiseed_{seeds[0]}_{seeds[-1]}.json"

    # Mean canonical metrics per (variant, scenario) across seeds
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}
    for summary in all_summaries:
        for run in summary.runs:
            key = (run.variant, run.scenario)
            bucket = cells.setdefault(key, {})
            for metric, value in run.metrics.items():
                if metric.startswith("latency_") or metric in CANONICAL_METRICS:
                    bucket.setdefault(metric, []).append(value)

    aggregated = []
    for (variant, scenario), metrics in sorted(cells.items()):
        row = {
            "variant": variant,
            "scenario": scenario,
            "seeds": seeds,
            "metrics": {k: mean(v) for k, v in metrics.items()},
            "metric_std": {
                k: (sum((x - mean(v)) ** 2 for x in v) / len(v)) ** 0.5 if len(v) > 1 else 0.0
                for k, v in metrics.items()
            },
        }
        aggregated.append(row)

    payload = {
        "seeds": seeds,
        "config_snapshot": cfg.public_summary(),
        "cells": aggregated,
        "per_seed_runs": len(all_summaries),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote multi-seed summary %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(description="Multi-seed ablation sweeps.")
    parser.add_argument("--seeds", default="42,43,44", help="comma-separated seeds")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_ABLATION_CONFIG)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    cfg_file = _load_ablation_config(args.config)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_multiseed(
        seeds,
        variants=cfg_file.get("variants"),
        scenarios=cfg_file.get("ablation_scenarios"),
        runs_dir=args.runs_dir,
        generate_plots=not args.no_plots,
    )
    print(f"Multi-seed sweep complete ({len(seeds)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
