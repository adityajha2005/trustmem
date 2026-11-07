"""Contradiction registry tests."""

from __future__ import annotations

from uuid import uuid4

from memory.contradictions import ContradictionRegistry, ResolutionStatus
from memory.lineage import LineageIndex
from memory.schema import Memory, MemoryScope


def test_resolve_conflicts_picks_latest_trusted() -> None:
    registry = ContradictionRegistry()
    m1 = Memory(content="deadline Monday", agent="planner", scope=MemoryScope.SHARED, confidence=0.7)
    m2 = Memory(
        content="deadline Friday",
        agent="verifier",
        scope=MemoryScope.SHARED,
        confidence=0.95,
    )
    index = LineageIndex.from_memories([m1, m2])
    claim = registry.register([m1.id, m2.id], detected_by="verifier")
    resolved = registry.resolve_conflicts(claim.id, index, requesting_agent="verifier")
    assert resolved.resolution_status == ResolutionStatus.RESOLVED
    assert resolved.winner_id == m2.id


def test_filter_for_retrieval_open_claim() -> None:
    registry = ContradictionRegistry()
    m1 = Memory(content="use REST", agent="coder", scope=MemoryScope.SHARED)
    m2 = Memory(content="use gRPC", agent="verifier", scope=MemoryScope.SHARED, confidence=0.9)
    memories = [m1, m2]
    index = LineageIndex.from_memories(memories)
    registry.register([m1.id, m2.id], detected_by="verifier")
    filtered, removed = registry.filter_for_retrieval(memories, index)
    assert removed >= 1
    assert len(filtered) == 1
    assert filtered[0].id == m2.id


def test_archived_memories_excluded_from_resolution() -> None:
    registry = ContradictionRegistry()
    old_id = uuid4()
    old = Memory(content="old fact", agent="a", scope=MemoryScope.SHARED, id=old_id, archived=True)
    new = Memory(content="new fact", agent="b", scope=MemoryScope.SHARED, supersedes=old_id)
    index = LineageIndex.from_memories([old, new])
    claim = registry.register([old.id, new.id])
    resolved = registry.resolve_conflicts(claim.id, index)
    assert resolved.winner_id == new.id
