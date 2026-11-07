"""Unit tests for retrieval scoring (no ML models)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memory.schema import Memory, MemoryScope
from retrieval.context import RetrievalContext
from retrieval.scoring import (
    ScoringWeights,
    compute_final_score,
    rank_memories,
    recency_score,
    trust_score,
)


def test_recency_decays_with_age() -> None:
    now = datetime.now(UTC)
    fresh = recency_score(now, reference=now)
    old = recency_score(now - timedelta(hours=168), reference=now, half_life_hours=168)
    assert fresh > old


def test_trust_self_authored() -> None:
    mem = Memory(content="x", agent="a", scope=MemoryScope.SHARED, confidence=0.3)
    ctx = RetrievalContext(agent_id="a", min_trust=0.5)
    assert trust_score(mem, ctx) == 1.0


def test_compute_final_score_weighted() -> None:
    score = compute_final_score(
        semantic=1.0,
        trust=1.0,
        recency=1.0,
        importance=1.0,
        weights=ScoringWeights(0.25, 0.25, 0.25, 0.25),
    )
    assert score == 1.0


def test_rank_memories_orders_by_semantic() -> None:
    now = datetime.now(UTC)
    m1 = Memory(content="alpha beta", agent="a", scope=MemoryScope.SHARED, task_id="t1")
    m2 = Memory(content="gamma delta", agent="a", scope=MemoryScope.SHARED, task_id="t1")
    ctx = RetrievalContext(agent_id="b", task_id="t1")
    ranked = rank_memories(
        [m1, m2],
        {str(m1.id): 0.9, str(m2.id): 0.1},
        ctx,
        reference_time=now,
    )
    assert ranked[0].memory.id == m1.id
