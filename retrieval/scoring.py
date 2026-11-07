"""Retrieval scoring: semantic, trust, recency, and importance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from memory.schema import Memory
from retrieval.context import RetrievalContext


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the final retrieval score (should sum to ~1.0)."""

    semantic: float = 0.45
    trust: float = 0.25
    recency: float = 0.15
    importance: float = 0.15


@dataclass(frozen=True)
class ScoredMemory:
    memory: Memory
    semantic_score: float
    trust_score: float
    recency_score: float
    importance_score: float
    final_score: float

    def as_hit(self) -> dict:
        m = self.memory
        return {
            "memory_id": str(m.id),
            "content": m.content,
            "scope": m.scope.value,
            "agent": m.agent,
            "task_id": m.task_id,
            "semantic_score": round(self.semantic_score, 6),
            "trust_score": round(self.trust_score, 6),
            "recency_score": round(self.recency_score, 6),
            "importance_score": round(self.importance_score, 6),
            "final_score": round(self.final_score, 6),
            "archived": m.archived,
        }


def trust_score(memory: Memory, ctx: RetrievalContext) -> float:
    """
    Trust signal from agent confidence and authorship.

    Self-authored memories score 1.0; shared memories use stored confidence.
    """
    if memory.agent == ctx.agent_id:
        return 1.0
    if memory.confidence < ctx.min_trust:
        return 0.0
    # TODO: Trust-aware retrieval — agent reputation, corroboration, decay
    return memory.confidence


def recency_score(
    timestamp: datetime,
    *,
    reference: datetime | None = None,
    half_life_hours: float = 168.0,
) -> float:
    """
    Exponential recency decay in [0, 1].

    TODO: Temporal reasoning — event windows, task phases, validity intervals.
    """
    ref = reference or datetime.now(UTC)
    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
    age_hours = max((ref - ts.astimezone(UTC)).total_seconds() / 3600.0, 0.0)
    if half_life_hours <= 0:
        return 1.0
    return math.exp(-0.693147 * age_hours / half_life_hours)


def compute_final_score(
    *,
    semantic: float,
    trust: float,
    recency: float,
    importance: float,
    weights: ScoringWeights,
) -> float:
    """
    Weighted linear combination of normalized signals.

    score = w_s·semantic + w_t·trust + w_r·recency + w_i·importance
    """
    return (
        weights.semantic * semantic
        + weights.trust * trust
        + weights.recency * recency
        + weights.importance * importance
    )


def rank_memories(
    memories: list[Memory],
    semantic_scores: dict[str, float],
    ctx: RetrievalContext,
    *,
    weights: ScoringWeights | None = None,
    reference_time: datetime | None = None,
    half_life_hours: float = 168.0,
) -> list[ScoredMemory]:
    """Score and sort memories by final_score descending."""
    w = weights or ScoringWeights()
    ref = reference_time or datetime.now(UTC)
    scored: list[ScoredMemory] = []

    for memory in memories:
        mem_id = str(memory.id)
        sem = semantic_scores.get(mem_id, 0.0)
        tr = trust_score(memory, ctx)
        if tr < ctx.min_trust:
            continue
        rec = recency_score(memory.timestamp, reference=ref, half_life_hours=half_life_hours)
        imp = memory.importance
        final = compute_final_score(
            semantic=sem,
            trust=tr,
            recency=rec,
            importance=imp,
            weights=w,
        )
        scored.append(
            ScoredMemory(
                memory=memory,
                semantic_score=sem,
                trust_score=tr,
                recency_score=rec,
                importance_score=imp,
                final_score=final,
            )
        )

    scored.sort(key=lambda s: s.final_score, reverse=True)
    return scored
