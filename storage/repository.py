"""Facade over SQLiteMemoryStore for retrieval and agent clients."""

from __future__ import annotations

from uuid import UUID

from memory.schema import Memory
from storage.sqlite_store import SQLiteMemoryStore


class MemoryRepository:
    """Thin wrapper preserving the repository name used by retrieval and API."""

    def __init__(self, session) -> None:  # noqa: ANN001
        self._store = SQLiteMemoryStore(session)

    @property
    def store(self) -> SQLiteMemoryStore:
        return self._store

    def add(self, memory: Memory) -> Memory:
        return self._store.create(memory)

    def get(self, memory_id: UUID) -> Memory | None:
        return self._store.get(memory_id)

    def list_by_task(
        self,
        task_id: str,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        return self._store.list_by_task(task_id, include_archived=include_archived, limit=limit)

    def list_for_agent(
        self,
        agent: str,
        *,
        task_id: str | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        return self._store.list_for_agent(
            agent, task_id=task_id, include_archived=include_archived, limit=limit
        )
