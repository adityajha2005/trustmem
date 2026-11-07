"""Experiment summary reports (Markdown + LaTeX-ready tables)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.ablation_metrics import CANONICAL_METRICS
from experiments.aggregate import (
    ExperimentSummary,
    build_comparison_table,
    compute_deltas,
    markdown_table,
)
from experiments.expected_failures import (
    EXPECTED_ABLATION_FAILURES,
    lookup_expected_failure,
    paper_ablation_discussion_paragraph,
)
from experiments.load import runs_dataframe
from utils.logging import get_logger

logger = get_logger(__name__)


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def comparison_to_latex(rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> str:
    """Simple LaTeX tabular for reports."""
    if not rows:
        return ""
    cols = columns or [c for c in rows[0] if not str(c).startswith("delta_")]
    header = " & ".join(_latex_escape(c) for c in cols) + r" \\"
    lines = [
        r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(_latex_escape(str(val)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def generate_summary_report(
    summary: ExperimentSummary,
    report_path: Path,
    *,
    plot_paths: dict[str, Path] | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Write a Markdown experiment report with tables and figure references.

    Also writes a sibling ``.tex`` snippet with the comparison table.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tag = run_id or f"ablation_seed{summary.seed}"

    comparison = build_comparison_table(summary, include_deltas=True)
    deltas = compute_deltas(summary)
    df = runs_dataframe(summary)

    sections: list[str] = [
        f"# Experiment report: {tag}",
        "",
        "## Reproducibility",
        "",
        f"- **Seed:** {summary.seed}",
        f"- **Baseline variant:** `{summary.baseline_variant}`",
        f"- **Git SHA:** `{summary.git_sha or 'n/a'}`",
        f"- **Cells:** {len(summary.runs)}",
        "",
        "## Summary",
        "",
        "Mean metrics per ablation variant (aggregated across benchmark scenarios). "
        "Lower contradiction and stale rates are better; higher precision, recall, "
        "and active-memory accuracy are better.",
        "",
        "## Expected failures (negative controls)",
        "",
        "Five (variant × scenario) cells **fail by design** on every seed. "
        "They are not bugs—do not tune retrieval to pass them. "
        "See `docs/ABLATION_FAILURES.md`.",
        "",
        paper_ablation_discussion_paragraph(),
        "",
    ]

    failed_runs = [r for r in summary.runs if not r.passed]
    if failed_runs:
        ef_rows: list[dict[str, str]] = []
        for run in failed_runs:
            spec = lookup_expected_failure(run.variant, run.scenario)
            ef_rows.append(
                {
                    "variant": run.variant,
                    "scenario": run.scenario,
                    "status": "expected" if spec else "UNEXPECTED",
                    "category": spec.category if spec else "—",
                    "paper_role": spec.paper_role if spec else "Investigate",
                }
            )
        sections.append(markdown_table(ef_rows))
        sections.append("")

    sections.extend(
        [
            "### Registered negative controls",
            "",
            markdown_table(
                [
                    {
                        "variant": e.variant,
                        "scenario": e.scenario,
                        "category": e.category,
                    }
                    for e in EXPECTED_ABLATION_FAILURES
                ]
            ),
            "",
            "## Comparison table",
            "",
        ]
    )

    metric_cols = ["variant", "scenarios", *CANONICAL_METRICS]
    delta_cols = [f"delta_{c}" for c in CANONICAL_METRICS if any(f"delta_{c}" in r for r in comparison)]
    md_cols = metric_cols + delta_cols
    sections.append(markdown_table(comparison, columns=md_cols))

    if deltas:
        sections.extend(
            [
                "",
                "## Per-variant deltas vs baseline",
                "",
            ]
        )
        for variant, delta_map in sorted(deltas.items()):
            notable = {k: v for k, v in delta_map.items() if abs(v) > 1e-4}
            if not notable:
                continue
            parts = ", ".join(f"`{k}`: {v:+.3f}" for k, v in sorted(notable.items())[:6])
            sections.append(f"- **{variant}:** {parts}")
        sections.append("")

    if not df.empty:
        breakdown = (
            df.groupby(["scenario", "variant"], sort=True)
            .agg(
                precision=("retrieval_precision", "mean"),
                recall=("retrieval_recall", "mean"),
                stale=("stale_retrieval_rate", "mean"),
                active=("active_memory_accuracy", "mean"),
            )
            .round(3)
            .reset_index()
        )
        rows = breakdown.to_dict(orient="records")
        cols = list(breakdown.columns)
        sections.extend(
            [
                "## Per-scenario breakdown",
                "",
                markdown_table(rows, columns=cols),
                "",
            ]
        )

    if plot_paths:
        sections.extend(["## Figures", ""])
        for name, png in sorted(plot_paths.items()):
            rel = _relative_path(report_path.parent, png)
            sections.append(f"### {name.replace('_', ' ').title()}")
            sections.append(f"![{name}]({rel})")
            sections.append("")

    report_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Wrote report %s", report_path)

    tex_path = report_path.with_suffix(".tex")
    tex_body = comparison_to_latex(comparison, columns=metric_cols)
    tex_path.write_text(
        f"% Auto-generated comparison table ({tag})\n{tex_body}\n",
        encoding="utf-8",
    )
    logger.info("Wrote LaTeX table %s", tex_path)

    return report_path


def _relative_path(from_dir: Path, target: Path) -> str:
    try:
        return Path(target).relative_to(from_dir).as_posix()
    except ValueError:
        return Path(target).as_posix()
