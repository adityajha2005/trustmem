"""Export ablation experiment artifacts (JSON, CSV, Markdown)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.aggregate import (
    ExperimentSummary,
    build_comparison_table,
    compute_deltas,
    latency_summary_table,
    markdown_table,
)
from evaluation.ablation_metrics import CANONICAL_METRICS, STAGE_LATENCY_KEYS
from utils.logging import get_logger

logger = get_logger(__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)


def export_experiment(
    summary: ExperimentSummary,
    out_dir: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    """
    Persist sweep artifacts under ``out_dir``.

    Returns paths written: summary.json, runs.csv, comparison.csv, deltas.json,
    comparison.md, latency.csv.
    """
    tag = run_id or f"ablation_seed{summary.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    summary_payload = {
        "seed": summary.seed,
        "baseline_variant": summary.baseline_variant,
        "git_sha": summary.git_sha,
        "run_count": len(summary.runs),
        "runs": [asdict(r) for r in summary.runs],
    }
    paths["summary"] = out_dir / f"{tag}_summary.json"
    _write_json(paths["summary"], summary_payload)

    paths["runs_csv"] = out_dir / f"{tag}_runs.csv"
    _write_csv(paths["runs_csv"], summary.rows())

    comparison = build_comparison_table(summary, include_deltas=True)
    paths["comparison_csv"] = out_dir / f"{tag}_comparison.csv"
    _write_csv(paths["comparison_csv"], comparison)

    deltas = compute_deltas(summary)
    paths["deltas"] = out_dir / f"{tag}_deltas.json"
    _write_json(paths["deltas"], deltas)

    metric_cols = ["variant", "scenarios", *CANONICAL_METRICS]
    delta_cols = [f"delta_{c}" for c in CANONICAL_METRICS]
    md_cols = metric_cols + [c for c in delta_cols if any(c in row for row in comparison)]
    paths["comparison_md"] = out_dir / f"{tag}_comparison.md"
    paths["comparison_md"].write_text(
        f"# Ablation comparison (seed={summary.seed})\n\n"
        + markdown_table(comparison, columns=md_cols),
        encoding="utf-8",
    )

    latency_rows = latency_summary_table(summary)
    if latency_rows:
        paths["latency_csv"] = out_dir / f"{tag}_latency.csv"
        lat_cols = ["variant"] + [c for c in STAGE_LATENCY_KEYS if c in latency_rows[0]]
        _write_csv(paths["latency_csv"], [{k: row.get(k, "") for k in lat_cols} for row in latency_rows])

    return paths


def export_experiment_with_artifacts(
    summary: ExperimentSummary,
    out_dir: Path,
    *,
    run_id: str | None = None,
    generate_plots: bool = True,
    plots_root: Path | None = None,
    reports_root: Path | None = None,
) -> dict[str, Path]:
    """
    Export CSV/JSON/Markdown tables and optionally figures + summary report.

    Figures land in ``experiments/plots/<run_id>/``; reports in
    ``experiments/reports/<run_id>_report.md``.
    """
    tag = run_id or f"ablation_seed{summary.seed}"
    paths = export_experiment(summary, out_dir, run_id=tag)
    if not generate_plots:
        return paths

    try:
        from experiments.visualize import generate_figures_and_report

        viz_paths = generate_figures_and_report(
            summary,
            run_id=tag,
            plots_dir=plots_root,
            reports_dir=reports_root,
        )
        paths.update(viz_paths)
    except ImportError as exc:
        logger.warning(
            "Plot generation skipped (install viz extras: pip install -e '.[viz]'): %s",
            exc,
        )
    return paths
