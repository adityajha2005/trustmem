"""Memory scope and access rule tests."""

from __future__ import annotations

from memory.schema import Memory, MemoryScope
from memory.scopes import can_read, promote_scope


def test_local_memory_only_visible_to_owner() -> None:
    record = Memory(content="secret plan", agent="agent_a", scope=MemoryScope.LOCAL)
    assert can_read(record, "agent_a") is True
    assert can_read(record, "agent_b") is False


def test_shared_memory_visible_to_any_agent() -> None:
    record = Memory(content="team fact", agent="agent_a", scope=MemoryScope.SHARED)
    assert can_read(record, "agent_b") is True


def test_promote_scope() -> None:
    record = Memory(content="x", agent="a", scope=MemoryScope.LOCAL)
    promoted = promote_scope(record, MemoryScope.SHARED)
    assert promoted.scope == MemoryScope.SHARED
