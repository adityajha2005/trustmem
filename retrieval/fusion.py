"""Score fusion for multi-source collaborative candidates."""

from __future__ import annotations

from typing import Any


def weighted_linear_fusion(
    candidates: list[dict[str, Any]],
    *,
    semantic_key: str = "semantic_score",
    trust_key: str = "trust_score",
    semantic_weight: float = 0.6,
    trust_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Fuse semantic similarity with trust — not RRF over FTS+vector lists.

    Research note: keep fusion simple and interpretable for ablations.
    TODO: Learned fusion / calibration on benchmark splits.
    """
    fused: list[dict[str, Any]] = []
    for item in candidates:
        sem = float(item.get(semantic_key, 0.0))
        trust = float(item.get(trust_key, 0.0))
        score = semantic_weight * sem + trust_weight * trust
        entry = dict(item)
        entry["fused_score"] = score
        fused.append(entry)
    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused
