"""Fuse local (agent-scoped) and shared (collaborative) retrieval pools."""

from __future__ import annotations

from memory.schema import Memory, MemoryScope
from utils.logging import get_logger

logger = get_logger(__name__)


def split_by_scope(memories: list[Memory]) -> tuple[list[Memory], list[Memory]]:
    """Partition candidates into local and shared pools."""
    local = [m for m in memories if m.scope == MemoryScope.LOCAL]
    shared = [m for m in memories if m.scope == MemoryScope.SHARED]
    return local, shared


def fuse_pools(
    local_hits: list[tuple[str, float]],
    shared_hits: list[tuple[str, float]],
    *,
    local_boost: float = 1.05,
) -> dict[str, float]:
    """
    Merge semantic scores from local and shared FAISS searches.

    Local memories receive a small boost when the requesting agent owns them,
    reflecting higher relevance of private notes for that agent.

    TODO: Hybrid BM25 + vector retrieval — add sparse channel per pool.
    """
    fused: dict[str, float] = {}
    for mem_id, score in shared_hits:
        fused[mem_id] = max(fused.get(mem_id, 0.0), score)
    for mem_id, score in local_hits:
        boosted = score * local_boost
        fused[mem_id] = max(fused.get(mem_id, 0.0), boosted)
    logger.debug("Fused %d local + %d shared -> %d ids", len(local_hits), len(shared_hits), len(fused))
    return fused
