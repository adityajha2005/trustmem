"""Backward-compatible re-exports; prefer memory.schema for new code."""

from memory.schema import Memory, MemoryScope, MemoryType, utc_now

# Legacy alias used by orchestration and older modules
MemoryRecord = Memory

__all__ = ["Memory", "MemoryRecord", "MemoryScope", "MemoryType", "utc_now"]
