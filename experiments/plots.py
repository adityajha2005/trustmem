"""Publication-style experiment figures (PNG + PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.ablation_metrics import CANONICAL_METRICS, STAGE_LATENCY_KEYS
from experiments.aggregate import (
    ExperimentSummary,
    aggregate_by_variant,
    build_comparison_table,
    latency_summary_table,
)
from experiments.load import runs_dataframe
from experiments.style import apply_publication_style, variant_color
from utils.logging import get_logger

logger = get_logger(__name__)

# Display labels for axes
METRIC_LABELS: dict[str, str] = {
    "retrieval_precision": "Retrieval precision",
    "retrieval_recall": "Retrieval recall",
    "contradiction_rate": "Contradiction rate",
    "stale_retrieval_rate": "Stale retrieval rate",
    "active_memory_accuracy": "Active memory accuracy",
    "ranking_inversion_rate": "Ranking inversion rate",
    "long_horizon_consistency": "Long-horizon consistency",
    "context_efficiency": "Context efficiency",
    "lineage_completeness": "Lineage completeness",
    "contradiction_resolution_rate": "Contradiction resolution rate",
}

STAGE_LABELS: dict[str, str] = {
    "latency_candidate_pool_ms": "Candidate pool",
    "latency_scope_filter_ms": "Scope filter",
    "latency_archive_filter_ms": "Archive filter",
    "latency_lineage_filter_ms": "Lineage filter",
    "latency_hybrid_search_ms": "Hybrid search",
    "latency_flat_search_ms": "Flat search",
    "latency_score_rank_ms": "Score rank",
    "latency_rerank_ms": "Rerank",
    "latency_total_ms": "Total",
}


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_publication_style()
    return plt


def save_figure(fig, stem: Path) -> dict[str, Path]:
    """Write PNG and PDF for a figure stem (path without suffix)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": stem.with_suffix(".png"),
        "pdf": stem.with_suffix(".pdf"),
    }
    fig.savefig(paths["png"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("Wrote %s and %s", paths["png"], paths["pdf"])
    return paths


def plot_ablation_comparison(
    summary: ExperimentSummary,
    out_dir: Path,
    *,
    metrics: tuple[str, ...] = (
        "retrieval_precision",
        "active_memory_accuracy",
        "stale_retrieval_rate",
        "long_horizon_consistency",
    ),
) -> dict[str, Path]:
    """Grouped bar chart: mean canonical metrics per ablation variant."""
    plt = _require_matplotlib()
    comparison = build_comparison_table(summary, metrics=metrics, include_deltas=False)
    if not comparison:
        return {}

    variants = [row["variant"] for row in comparison]
    x = range(len(variants))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, metric in enumerate(metrics):
        offsets = [xi + (i - len(metrics) / 2) * width + width / 2 for xi in x]
        values = [row.get(metric, 0.0) for row in comparison]
        ax.bar(
            offsets,
            values,
            width=width,
            label=METRIC_LABELS.get(metric, metric),
            color=[variant_color(v, i) for v in variants],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation comparison (mean across scenarios)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    return save_figure(fig, out_dir / "ablation_comparison")


def plot_contradiction_rate(summary: ExperimentSummary, out_dir: Path) -> dict[str, Path]:
    """Contradiction rate by variant and scenario."""
    plt = _require_matplotlib()
    df = runs_dataframe(summary)
    if df.empty or "contradiction_rate" not in df.columns:
        return {}

    pivot = df.pivot_table(
        index="variant",
        columns="scenario",
        values="contradiction_rate",
        aggfunc="mean",
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    scenarios = list(pivot.columns)
    variants = list(pivot.index)
    x = range(len(variants))
    width = 0.8 / max(len(scenarios), 1)

    for i, scenario in enumerate(scenarios):
        offsets = [xi + (i - len(scenarios) / 2) * width + width / 2 for xi in x]
        ax.bar(
            offsets,
            pivot[scenario].values,
            width=width,
            label=scenario.replace("_v1", ""),
            alpha=0.9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(METRIC_LABELS["contradiction_rate"])
    ax.set_title("Contradiction rate by scenario")
    ax.legend(title="Scenario", frameon=False, fontsize=8)
    return save_figure(fig, out_dir / "contradiction_rate")


def plot_long_horizon_consistency(summary: ExperimentSummary, out_dir: Path) -> dict[str, Path]:
    """Long-horizon consistency across ablation variants."""
    plt = _require_matplotlib()
    df = runs_dataframe(summary)
    if df.empty:
        return {}

    col = "long_horizon_consistency"
    subset = df[df["scenario"].str.contains("long_horizon", na=False)]
    if subset.empty:
        subset = df
    if col not in subset.columns:
        return {}

    grouped = subset.groupby("variant", sort=True)[col].mean().sort_index()
    variants = list(grouped.index)
    values = grouped.values

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [variant_color(v, i) for i, v in enumerate(variants)]
    ax.bar(variants, values, color=colors, alpha=0.9, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(METRIC_LABELS[col])
    ax.set_title("Long-horizon consistency")
    ax.tick_params(axis="x", rotation=35)
    return save_figure(fig, out_dir / "long_horizon_consistency")


def plot_precision_recall(summary: ExperimentSummary, out_dir: Path) -> dict[str, Path]:
    """Precision vs recall per variant (mean across scenarios)."""
    plt = _require_matplotlib()
    by_variant = aggregate_by_variant(summary)
    if not by_variant:
        return {}

    variants = sorted(by_variant.keys(), key=lambda v: (v != summary.baseline_variant, v))
    precision = [by_variant[v].metrics.get("retrieval_precision", 0.0) for v in variants]
    recall = [by_variant[v].metrics.get("retrieval_recall", 0.0) for v in variants]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(variants))
    w = 0.35
    ax.bar([i - w / 2 for i in x], precision, width=w, label="Precision", color="#2E86AB", alpha=0.9)
    ax.bar([i + w / 2 for i in x], recall, width=w, label="Recall", color="#E94F37", alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval precision and recall")
    ax.legend(frameon=False)
    return save_figure(fig, out_dir / "precision_recall")


def plot_stage_latency(summary: ExperimentSummary, out_dir: Path) -> dict[str, Path]:
    """Stacked bar: mean stage latency (ms) per variant."""
    plt = _require_matplotlib()
    rows = latency_summary_table(summary)
    if not rows:
        return {}

    import pandas as pd

    df = pd.DataFrame(rows).set_index("variant")
    stage_cols = [c for c in STAGE_LATENCY_KEYS if c in df.columns and c != "latency_total_ms"]
    if not stage_cols:
        return {}

    # Drop all-zero columns for cleaner plots
    stage_cols = [c for c in stage_cols if df[c].sum() > 0]
    if not stage_cols:
        return {}

    variants = list(df.index)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0.0] * len(variants)
    try:
        cmap_obj = plt.colormaps["tab20"].resampled(max(len(stage_cols), 1))
        color_at = lambda i: cmap_obj(i / max(len(stage_cols) - 1, 1))
    except AttributeError:
        cmap_legacy = plt.cm.get_cmap("tab20", len(stage_cols))

        def color_at(i: int):  # noqa: ANN202
            return cmap_legacy(i)

    for i, col in enumerate(stage_cols):
        values = df[col].fillna(0.0).values
        ax.bar(
            variants,
            values,
            bottom=bottom,
            label=STAGE_LABELS.get(col, col),
            color=color_at(i),
            alpha=0.92,
            edgecolor="white",
            linewidth=0.3,
        )
        bottom = [b + v for b, v in zip(bottom, values, strict=True)]

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Retrieval stage latency (stacked mean)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    return save_figure(fig, out_dir / "stage_latency")


def plot_lineage_suppression(summary: ExperimentSummary, out_dir: Path) -> dict[str, Path]:
    """
    Lineage filter effectiveness: stale rate vs active accuracy.

    Highlights ``full`` vs ``no_lineage_filter`` when present.
    """
    plt = _require_matplotlib()
    df = runs_dataframe(summary)
    if df.empty:
        return {}

    focus = df[df["variant"].isin(["full", "no_lineage_filter"])]
    if focus.empty:
        focus = df

    grouped = focus.groupby("variant", sort=True).agg(
        stale_retrieval_rate=("stale_retrieval_rate", "mean"),
        active_memory_accuracy=("active_memory_accuracy", "mean"),
    )

    variants = list(grouped.index)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True)

    for ax, metric, title in zip(
        axes,
        ["stale_retrieval_rate", "active_memory_accuracy"],
        ["Stale retrieval (lower is better)", "Active memory accuracy"],
        strict=True,
    ):
        colors = [variant_color(v, i) for i, v in enumerate(variants)]
        ax.bar(variants, grouped[metric].values, color=colors, alpha=0.9, edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Lineage suppression effectiveness", fontsize=11, y=1.02)
    return save_figure(fig, out_dir / "lineage_suppression")


def generate_all_plots(summary: ExperimentSummary, plots_dir: Path) -> dict[str, Path]:
    """Generate the full figure set; returns map of logical name → PNG path."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_paths: dict[str, Path] = {}
    generators = [
        ("ablation_comparison", plot_ablation_comparison),
        ("contradiction_rate", plot_contradiction_rate),
        ("long_horizon_consistency", plot_long_horizon_consistency),
        ("precision_recall", plot_precision_recall),
        ("stage_latency", plot_stage_latency),
        ("lineage_suppression", plot_lineage_suppression),
    ]
    for name, fn in generators:
        try:
            paths = fn(summary, plots_dir)
            if paths:
                all_paths[name] = paths["png"]
        except Exception as exc:
            logger.warning("Plot %s skipped: %s", name, exc)
    return all_paths
