"""Facade for lineage-aware memory evolution workflows."""

from __future__ import annotations

from uuid import UUID

from memory.contradictions import ContradictionClaim, ContradictionRegistry, ResolutionStatus
from memory.lineage import LineageIndex, filter_lineage_active
from memory.schema import Memory
from storage.sqlite_store import SQLiteMemoryStore
from utils.logging import get_logger

logger = get_logger(__name__)


class MemoryEvolution:
    """
    High-level API for planner / coder / verifier memory workflows.

    Example:
        planner writes plan → coder supersedes with implementation
        → verifier registers contradiction → resolve_conflicts picks winner
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        registry: ContradictionRegistry | None = None,
    ) -> None:
        self._store = store
        self.registry = registry or ContradictionRegistry()

    def _load_task(self, task_id: str, *, include_archived: bool = True) -> list[Memory]:
        return self._store.list_by_task(task_id, include_archived=include_archived)

    def _index(self, memories: list[Memory]) -> LineageIndex:
        return LineageIndex.from_memories(memories)

    def get_memory_lineage(self, memory_id: UUID, *, task_id: str | None = None) -> list[Memory]:
        memories = self._load_scope(memory_id, task_id)
        return self._index(memories).get_memory_lineage(memory_id)

    def get_latest_memory(self, memory_id: UUID, *, task_id: str | None = None) -> Memory | None:
        memories = self._load_scope(memory_id, task_id)
        return self._index(memories).get_latest_memory(memory_id)

    def get_active_memories(self, task_id: str, *, include_archived: bool = False) -> list[Memory]:
        memories = self._load_task(task_id, include_archived=include_archived)
        active = self._index(memories).get_active_memories(include_archived=include_archived)
        filtered, _ = self.registry.filter_for_retrieval(
            active, self._index(memories), include_archived=include_archived
        )
        return filtered

    def resolve_conflicts(
        self,
        claim_id: UUID,
        *,
        winner_id: UUID | None = None,
        requesting_agent: str | None = None,
        archive_losers: bool = True,
        task_id: str | None = None,
    ) -> ContradictionClaim:
        """
        Resolve a contradiction and optionally archive losing revisions.

        Losers are archived via the store; winners remain active for retrieval.
        """
        claim = self.registry.get(claim_id)
        if claim is None:
            raise KeyError(f"Unknown claim: {claim_id}")

        memories = self._load_scope(claim.memory_ids[0], task_id)
        lineage = self._index(memories)
        resolved = self.registry.resolve_conflicts(
            claim_id,
            lineage,
            winner_id=winner_id,
            requesting_agent=requesting_agent,
        )

        if archive_losers and resolved.winner_id is not None:
            for mem_id in claim.memory_ids:
                if mem_id == resolved.winner_id:
                    continue
                latest = lineage.get_latest_memory(mem_id)
                if latest is not None and latest.id != resolved.winner_id:
                    self._store.archive(latest.id)

        return resolved

    def register_contradiction(
        self,
        memory_ids: list[UUID],
        *,
        detected_by: str | None = None,
        confidence: float = 0.5,
        notes: str = "",
    ) -> ContradictionClaim:
        return self.registry.register(
            memory_ids, detected_by=detected_by, confidence=confidence, notes=notes
        )

    def supersede(
        self,
        old_id: UUID,
        new_memory: Memory,
        *,
        archive_old: bool = True,
    ) -> Memory:
        """Create a successor memory and optionally archive the predecessor."""
        if archive_old:
            return self._store.supersede(old_id, new_memory)
        new_memory = new_memory.model_copy(
            update={"supersedes": old_id, "parent_memory": new_memory.parent_memory or old_id}
        )
        return self._store.create(new_memory)

    def retrieval_candidates(
        self,
        task_id: str,
        *,
        include_archived: bool = False,
    ) -> list[Memory]:
        """Memories safe for retrieval: active lineage + contradiction filtering."""
        memories = self._load_task(task_id, include_archived=True)
        lineage = self._index(memories)
        active = filter_lineage_active(memories, include_archived=include_archived)
        return self.registry.filter_for_retrieval(
            active, lineage, include_archived=include_archived
        )[0]

    def _load_scope(self, memory_id: UUID, task_id: str | None) -> list[Memory]:
        memory = self._store.get(memory_id)
        if memory is None:
            return []
        tid = task_id or memory.task_id
        if tid:
            return self._load_task(tid, include_archived=True)
        return [memory]
