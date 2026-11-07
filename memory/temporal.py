"""Temporal helpers for active vs stale memory resolution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memory.schema import Memory
from memory.lineage import LineageIndex

# TODO: Temporal memory graphs — task phases, validity intervals, event ordering


def utc_now() -> datetime:
    return datetime.now(UTC)


def is_stale_by_age(
    memory: Memory,
    *,
    reference: datetime | None = None,
    max_age: timedelta | None = None,
) -> bool:
    """Mark memory temporally stale when older than max_age."""
    if max_age is None:
        return False
    ref = reference or utc_now()
    ts = memory.timestamp if memory.timestamp.tzinfo else memory.timestamp.replace(tzinfo=UTC)
    return (ref - ts.astimezone(UTC)) > max_age


def is_active_state(
    memory: Memory,
    lineage: LineageIndex,
    *,
    reference: datetime | None = None,
    max_age: timedelta | None = None,
) -> bool:
    """
    Combined lineage + temporal active check.

    Active when: not superseded, not archived, and within optional age window.
    """
    if lineage.is_stale(memory):
        return False
    if is_stale_by_age(memory, reference=reference, max_age=max_age):
        return False
    return True


def prefer_latest_trusted(
    candidates: list[Memory],
    *,
    requesting_agent: str | None = None,
) -> Memory | None:
    """
    Pick the best candidate by recency and confidence.

    Self-authored memories win ties; used during conflict resolution.
    TODO: Adaptive trust evolution — agent reputation over time.
    """
    if not candidates:
        return None

    def sort_key(m: Memory) -> tuple[float, float, float]:
        agent_boost = 1.0 if requesting_agent and m.agent == requesting_agent else 0.0
        return (agent_boost, m.confidence, m.timestamp.timestamp())

    return max(candidates, key=sort_key)


def sort_by_recency(memories: list[Memory], *, descending: bool = True) -> list[Memory]:
    return sorted(memories, key=lambda m: m.timestamp, reverse=descending)
