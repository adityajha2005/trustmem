"""Scope access rules for local vs shared memory."""

from __future__ import annotations

from memory.schema import Memory, MemoryScope


def can_read(record: Memory, agent_id: str) -> bool:
    """
    Decide whether an agent may read a memory.

    Shared memories are readable by any agent; local memories only by the author.
    """
    if record.scope == MemoryScope.SHARED:
        return True
    return record.agent == agent_id


def promote_scope(record: Memory, target: MemoryScope) -> Memory:
    """Return a copy with elevated scope (local → shared)."""
    if target == MemoryScope.LOCAL:
        return record
    if record.scope == MemoryScope.SHARED:
        return record
    return record.model_copy(update={"scope": MemoryScope.SHARED})
