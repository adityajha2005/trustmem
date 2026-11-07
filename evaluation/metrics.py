"""Metric primitives for collaborative memory benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollaborativeMetrics:
    """Outcome metrics for a benchmark run (legacy aggregate)."""

    recall_at_k: float = 0.0
    cross_agent_consistency: float = 0.0
    contradiction_resolution_rate: float = 0.0
    lineage_completeness: float = 0.0
    trust_calibration_error: float = 0.0


# ---------------------------------------------------------------------------
# Set-based retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    if k <= 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of relevant items that appear in top-k retrieved."""
    if not relevant or k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def f1_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    p = precision_at_k(relevant, retrieved, k)
    r = recall_at_k(relevant, retrieved, k)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Memory-evolution metrics
# ---------------------------------------------------------------------------


def stale_retrieval_rate(retrieved: list[str], stale: set[str]) -> float:
    """Share of retrieved hits that are archived or superseded."""
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in stale) / len(retrieved)


def active_memory_accuracy(retrieved: list[str], active: set[str]) -> float:
    """Share of retrieved hits that are part of the current active set."""
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in active) / len(retrieved)


def context_efficiency(retrieved: list[str], useful: set[str], k: int) -> float:
    """Useful results within the top-k budget (1.0 = every slot used productively)."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in useful) / k


def long_horizon_consistency(per_query_active_accuracy: list[float]) -> float:
    """Average active-memory accuracy across an evolving session timeline."""
    if not per_query_active_accuracy:
        return 0.0
    return sum(per_query_active_accuracy) / len(per_query_active_accuracy)


def ranking_inversion_rate(
    ranked_ids: list[str],
    score_by_id: dict[str, float],
) -> float:
    """
    Fraction of ranked pairs out of order vs an oracle score (e.g. confidence).

    0.0 = perfectly calibrated, 1.0 = fully inverted.
    """
    n = len(ranked_ids)
    if n < 2:
        return 0.0
    inversions = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            si = score_by_id.get(ranked_ids[i], 0.0)
            sj = score_by_id.get(ranked_ids[j], 0.0)
            if si != sj:
                pairs += 1
                if si < sj:
                    inversions += 1
    return inversions / pairs if pairs else 0.0


def contradiction_resolution_rate(claim_outcomes: list[bool]) -> float:
    """Share of claims resolved with the expected winner."""
    if not claim_outcomes:
        return 0.0
    return sum(1 for ok in claim_outcomes if ok) / len(claim_outcomes)


def suppression_rate(suppressed: int, total_losers: int) -> float:
    """Share of contradiction losers that retrieval no longer surfaces."""
    if total_losers <= 0:
        return 0.0
    return min(suppressed, total_losers) / total_losers


# ---------------------------------------------------------------------------
# Rerank fusion / trust calibration metrics
# ---------------------------------------------------------------------------


def trust_preservation_rate(
    retrieved: list[str],
    trust_by_id: dict[str, float],
    k: int,
) -> float:
    """
    Fraction of the top-k items by trust that survive in the top-k retrieved.

    1.0 = all high-trust memories preserved at the head;
    0.0 = none of them appear in the top-k.
    """
    if k <= 0 or not retrieved or not trust_by_id:
        return 1.0
    sorted_by_trust = sorted(
        trust_by_id.keys(),
        key=lambda mem_id: trust_by_id[mem_id],
        reverse=True,
    )
    top_trusted = set(sorted_by_trust[: min(k, len(sorted_by_trust))])
    if not top_trusted:
        return 1.0
    retrieved_top = set(retrieved[: min(k, len(retrieved))])
    return len(top_trusted & retrieved_top) / len(top_trusted)


def rerank_disruption_rate(
    pre: list[str],
    post: list[str],
    k: int | None = None,
) -> float:
    """
    Fraction of pre-rerank top-k positions that changed identity post-rerank.

    0.0 = rerank preserved the head; 1.0 = entirely re-ordered.
    """
    if not pre or not post:
        return 0.0
    n = min(len(pre), len(post))
    if k is not None:
        n = min(n, k)
    if n == 0:
        return 0.0
    moved = sum(1 for i in range(n) if pre[i] != post[i])
    return moved / n


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
