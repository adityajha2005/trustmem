"""Orchestrate plot generation and experiment reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.aggregate import ExperimentSummary
from experiments.load import load_summary_from_run
from experiments.plots import generate_all_plots
from experiments.reports import generate_summary_report
from utils.config import PROJECT_ROOT
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_PLOTS_ROOT = PROJECT_ROOT / "experiments" / "plots"
DEFAULT_REPORTS_ROOT = PROJECT_ROOT / "experiments" / "reports"


def generate_figures_and_report(
    summary: ExperimentSummary,
    *,
    run_id: str,
    plots_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Generate all figures (PNG/PDF) and a Markdown summary report.

    Returns paths: plot stems, ``report``, ``report_tex``.
    """
    plots_dir = Path(plots_dir or DEFAULT_PLOTS_ROOT / run_id)
    reports_dir = Path(reports_dir or DEFAULT_REPORTS_ROOT)

    plot_paths = generate_all_plots(summary, plots_dir)
    report_path = reports_dir / f"{run_id}_report.md"
    generate_summary_report(
        summary,
        report_path,
        plot_paths=plot_paths,
        run_id=run_id,
    )

    out: dict[str, Path] = dict(plot_paths)
    out["report"] = report_path
    out["report_tex"] = report_path.with_suffix(".tex")
    return out


def generate_from_run_dir(
    run_dir: Path,
    *,
    run_id: str | None = None,
    plots_root: Path | None = None,
    reports_root: Path | None = None,
) -> dict[str, Path]:
    """Load artifacts from a prior sweep and regenerate plots + report."""
    summary = load_summary_from_run(run_dir, run_id)
    tag = run_id or f"ablation_seed{summary.seed}"
    return generate_figures_and_report(
        summary,
        run_id=tag,
        plots_dir=(plots_root or DEFAULT_PLOTS_ROOT) / tag,
        reports_dir=reports_root or DEFAULT_REPORTS_ROOT,
    )


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(description="Generate experiment plots and reports.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing *_summary.json (e.g. experiments/runs/ablations/ablation_seed42)",
    )
    parser.add_argument("--run-id", default=None, help="Run tag matching summary filename prefix")
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    paths = generate_from_run_dir(
        args.run_dir,
        run_id=args.run_id,
        plots_root=args.plots_dir,
        reports_root=args.reports_dir,
    )
    print(f"Generated {len(paths)} artifacts:")
    for name, path in sorted(paths.items()):
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
