"""Publication-style matplotlib defaults (deterministic)."""

from __future__ import annotations

# Fixed palette: stable color per variant name (hash order, not random).
VARIANT_COLORS: dict[str, str] = {
    "full": "#2E86AB",
    "no_lineage_filter": "#E94F37",
    "no_trust_scoring": "#F6AE2D",
    "no_archive_filter": "#A23B72",
    "no_reranker": "#6B705C",
    "local_only": "#168AAD",
    "shared_only": "#BC4749",
    "flat_baseline": "#7F7F7F",
    "collaborative": "#2E86AB",
    "simple_rag": "#F6AE2D",
    "mem0": "#E94F37",
}

FALLBACK_COLORS = [
    "#2E86AB",
    "#E94F37",
    "#F6AE2D",
    "#A23B72",
    "#6B705C",
    "#168AAD",
    "#BC4749",
    "#7F7F7F",
]


def variant_color(variant: str, index: int = 0) -> str:
    if variant in VARIANT_COLORS:
        return VARIANT_COLORS[variant]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def apply_publication_style() -> None:
    """Configure matplotlib for report-ready figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "figure.constrained_layout.use": True,
        }
    )
