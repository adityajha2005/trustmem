"""Publication figures comparing memory techniques (mem0, RAG, collaborative)."""

from __future__ import annotations

from pathlib import Path

from baselines.techniques import technique_label
from evaluation.ablation_metrics import CANONICAL_METRICS
from experiments.aggregate import ExperimentSummary, aggregate_by_variant
from experiments.plots import METRIC_LABELS, save_figure, _require_matplotlib
from experiments.style import apply_publication_style
from utils.config import PROJECT_ROOT
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PLOTS_ROOT = PROJECT_ROOT / "experiments" / "plots"

TECHNIQUE_COLORS: dict[str, str] = {
    "collaborative": "#2E86AB",
    "simple_rag": "#F6AE2D",
    "mem0": "#E94F37",
}

COMPARISON_METRICS: tuple[str, ...] = (
    "retrieval_precision",
    "retrieval_recall",
    "active_memory_accuracy",
    "long_horizon_consistency",
    "stale_retrieval_rate",
    "contradiction_resolution_rate",
)


def technique_color(technique: str, index: int = 0) -> str:
    return TECHNIQUE_COLORS.get(technique, "#7F7F7F")


def plot_technique_overview(
    summary: ExperimentSummary,
    out_dir: Path,
) -> dict[str, Path]:
    """Grouped bar chart: mean metrics per technique."""
    plt = _require_matplotlib()
    grouped = aggregate_by_variant(summary)
    if not grouped:
        return {}

    techniques = sorted(
        grouped.keys(),
        key=lambda t: (t != "collaborative", t),
    )
    sample_metrics = next(iter(grouped.values())).metrics
    metrics = [m for m in COMPARISON_METRICS if m in sample_metrics]
    if not metrics:
        metrics = list(CANONICAL_METRICS[:4])

    x = range(len(techniques))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(metrics):
        offsets = [xi + (i - len(metrics) / 2) * width + width / 2 for xi in x]
        values = [float(grouped[t].metrics.get(metric, 0.0)) for t in techniques]
        ax.bar(
            offsets,
            values,
            width=width,
            label=METRIC_LABELS.get(metric, metric),
            color=[technique_color(t, i) for t in techniques],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([technique_label(t) for t in techniques], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Memory technique benchmark (mean across scenarios)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
    return save_figure(fig, out_dir / "technique_overview")


def plot_technique_by_scenario(
    summary: ExperimentSummary,
    out_dir: Path,
    *,
    metric: str = "retrieval_precision",
) -> dict[str, Path]:
    """Heatmap-style grouped bars: technique × scenario for one metric."""
    plt = _require_matplotlib()
    import pandas as pd

    from experiments.load import runs_dataframe

    df = runs_dataframe(summary)
    if df.empty or metric not in df.columns:
        return {}

    pivot = df.pivot_table(index="variant", columns="scenario", values=metric, aggfunc="mean")
    techniques = sorted(pivot.index.tolist(), key=lambda t: (t != "collaborative", t))
    scenarios = list(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.2), 4.5))
    x = range(len(scenarios))
    width = 0.8 / max(len(techniques), 1)

    for i, technique in enumerate(techniques):
        offsets = [xi + (i - len(techniques) / 2) * width + width / 2 for xi in x]
        values = [float(pivot.loc[technique, s]) for s in scenarios]
        ax.bar(
            offsets,
            values,
            width=width,
            label=technique_label(technique),
            color=technique_color(technique, i),
            alpha=0.9,
            edgecolor="white",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by technique and scenario")
    ax.legend(frameon=False)
    return save_figure(fig, out_dir / f"technique_by_scenario_{metric}")


def plot_technique_radar(
    summary: ExperimentSummary,
    out_dir: Path,
) -> dict[str, Path]:
    """Radar chart comparing techniques on core metrics."""
    plt = _require_matplotlib()
    import numpy as np

    grouped = aggregate_by_variant(summary)
    if not grouped:
        return {}

    radar_metrics = (
        "retrieval_precision",
        "retrieval_recall",
        "active_memory_accuracy",
        "long_horizon_consistency",
    )
    sample = next(iter(grouped.values())).metrics
    metrics = [m for m in radar_metrics if m in sample]
    if not metrics:
        return {}

    techniques = sorted(grouped.keys(), key=lambda t: (t != "collaborative", t))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for technique in techniques:
        values = [float(grouped[technique].metrics.get(m, 0.0)) for m in metrics]
        values += values[:1]
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=technique_label(technique),
            color=technique_color(technique),
        )
        ax.fill(angles, values, alpha=0.12, color=technique_color(technique))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], size=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Technique profile (radar)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False)
    return save_figure(fig, out_dir / "technique_radar")


def plot_pass_rate(summary: ExperimentSummary, out_dir: Path) -> dict[str, Path]:
    """Bar chart: fraction of scenarios passed per technique."""
    plt = _require_matplotlib()
    import pandas as pd

    from experiments.load import runs_dataframe

    df = runs_dataframe(summary)
    if df.empty:
        return {}

    rates = (
        df.groupby("variant")["passed"]
        .mean()
        .reindex(sorted(df["variant"].unique(), key=lambda t: (t != "collaborative", t)))
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    techniques = list(rates.index)
    colors = [technique_color(t, i) for i, t in enumerate(techniques)]
    ax.bar(
        [technique_label(t) for t in techniques],
        rates.values,
        color=colors,
        alpha=0.9,
        edgecolor="white",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pass rate")
    ax.set_title("Scenario pass rate by technique")
    ax.tick_params(axis="x", rotation=12)
    return save_figure(fig, out_dir / "technique_pass_rate")


def generate_technique_plots(
    summary: ExperimentSummary,
    run_id: str,
    *,
    plots_root: Path | None = None,
) -> dict[str, Path]:
    """Generate full technique comparison figure set."""
    apply_publication_style()
    plots_dir = Path(plots_root or DEFAULT_PLOTS_ROOT) / run_id
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_paths: dict[str, Path] = {}
    generators = [
        ("technique_overview", plot_technique_overview),
        ("technique_radar", plot_technique_radar),
        ("technique_pass_rate", plot_pass_rate),
    ]
    for name, fn in generators:
        try:
            paths = fn(summary, plots_dir)
            if paths:
                all_paths[name] = paths["png"]
        except Exception as exc:
            logger.warning("Plot %s skipped: %s", name, exc)

    for metric in ("retrieval_precision", "retrieval_recall", "active_memory_accuracy"):
        try:
            paths = plot_technique_by_scenario(summary, plots_dir, metric=metric)
            if paths:
                all_paths[f"by_scenario_{metric}"] = paths["png"]
        except Exception as exc:
            logger.warning("Plot by_scenario_%s skipped: %s", metric, exc)

    return all_paths
