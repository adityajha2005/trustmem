"""Load experiment summaries from ``experiments/runs/`` artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.ablation_metrics import AblationRunMetrics
from experiments.aggregate import ExperimentSummary


def summary_from_dict(data: dict) -> ExperimentSummary:
    runs = [
        AblationRunMetrics(
            variant=str(r["variant"]),
            scenario=str(r["scenario"]),
            metrics={k: float(v) for k, v in r.get("metrics", {}).items()},
            passed=bool(r.get("passed", True)),
            query_count=int(r.get("query_count", 0)),
        )
        for r in data.get("runs", [])
    ]
    return ExperimentSummary(
        seed=int(data.get("seed", 42)),
        baseline_variant=str(data.get("baseline_variant", "full")),
        runs=runs,
        git_sha=data.get("git_sha"),
    )


def load_summary_json(path: Path) -> ExperimentSummary:
    data = json.loads(path.read_text(encoding="utf-8"))
    return summary_from_dict(data)


def find_summary_file(run_dir: Path, run_id: str | None = None) -> Path:
    """
    Locate ``*_summary.json`` under ``run_dir``.

    If ``run_id`` is given, expects ``{run_id}_summary.json``.
    """
    run_dir = Path(run_dir)
    if run_id:
        candidate = run_dir / f"{run_id}_summary.json"
        if candidate.exists():
            return candidate
    matches = sorted(run_dir.glob("*_summary.json"))
    if not matches:
        raise FileNotFoundError(f"No summary JSON in {run_dir}")
    return matches[-1]


def load_summary_from_run(run_dir: Path, run_id: str | None = None) -> ExperimentSummary:
    return load_summary_json(find_summary_file(run_dir, run_id))


def runs_dataframe(summary: ExperimentSummary):
    """Pandas DataFrame of per-cell metrics (requires pandas)."""
    import pandas as pd

    return pd.DataFrame(summary.rows())
