"""Memory lineage tests."""

from __future__ import annotations

from uuid import uuid4

from memory.lineage import LineageIndex
from memory.schema import Memory, MemoryScope


def _mem(content: str, **kwargs) -> Memory:
    return Memory(content=content, agent="agent_a", scope=MemoryScope.SHARED, **kwargs)


def test_superseded_detection() -> None:
    old_id = uuid4()
    old = _mem("plan v1", id=old_id, archived=True)
    new = _mem("plan v2", supersedes=old_id, parent_memory=old_id)
    index = LineageIndex.from_memories([old, new])
    assert index.is_superseded(old_id) is True
    assert index.is_superseded(new.id) is False


def test_get_latest_memory() -> None:
    old_id = uuid4()
    old = _mem("v1", id=old_id, archived=True)
    mid = _mem("v2", supersedes=old_id, archived=True)
    latest = _mem("v3", supersedes=mid.id)
    index = LineageIndex.from_memories([old, mid, latest])
    tip = index.get_latest_memory(old_id)
    assert tip is not None and tip.id == latest.id


def test_get_active_memories_excludes_superseded() -> None:
    old_id = uuid4()
    old = _mem("stale", id=old_id, archived=True)
    new = _mem("current", supersedes=old_id)
    index = LineageIndex.from_memories([old, new])
    active = index.get_active_memories()
    assert len(active) == 1
    assert active[0].id == new.id


def test_get_memory_lineage() -> None:
    a_id = uuid4()
    a = _mem("a", id=a_id)
    b = _mem("b", supersedes=a_id, parent_memory=a_id)
    c = _mem("c", supersedes=b.id, parent_memory=b.id)
    chain = LineageIndex.from_memories([a, b, c]).get_memory_lineage(a_id)
    assert [m.id for m in chain] == [a_id, b.id, c.id]
