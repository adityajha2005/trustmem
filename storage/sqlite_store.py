"""SQLite-backed CRUD store for collaborative memory."""

from __future__ import annotations

from datetime import UTC
from uuid import UUID

from sqlalchemy import Select, and_, or_, select
from sqlalchemy.orm import Session

from memory.schema import Memory, MemoryScope, MemoryType, utc_now
from storage.models import MemoryModel
from utils.logging import get_logger

logger = get_logger(__name__)


def _row_to_memory(row: MemoryModel) -> Memory:
    return Memory(
        id=UUID(row.id),
        content=row.content,
        agent=row.agent,
        timestamp=row.timestamp,
        confidence=row.confidence,
        importance=row.importance,
        scope=MemoryScope(row.scope),
        task_id=row.task_id,
        memory_type=MemoryType(row.memory_type),
        parent_memory=UUID(row.parent_memory) if row.parent_memory else None,
        supersedes=UUID(row.supersedes) if row.supersedes else None,
        archived=row.archived,
    )


def _memory_to_row(memory: Memory) -> MemoryModel:
    return MemoryModel(
        id=str(memory.id),
        content=memory.content,
        agent=memory.agent,
        timestamp=memory.timestamp.astimezone(UTC),
        confidence=memory.confidence,
        importance=memory.importance,
        scope=memory.scope.value,
        task_id=memory.task_id,
        memory_type=memory.memory_type.value,
        parent_memory=str(memory.parent_memory) if memory.parent_memory else None,
        supersedes=str(memory.supersedes) if memory.supersedes else None,
        archived=memory.archived,
    )


class SQLiteMemoryStore:
    """
    CRUD operations for local and shared memories.

    Local memories are scoped to a single agent; shared memories are visible
    to all agents working on the same task (when task_id is set).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, memory: Memory) -> Memory:
        """Insert a new memory."""
        row = _memory_to_row(memory)
        self._session.add(row)
        self._session.flush()
        logger.debug("Created memory %s scope=%s agent=%s", memory.id, memory.scope, memory.agent)
        return _row_to_memory(row)

    def get(self, memory_id: UUID) -> Memory | None:
        """Fetch a memory by id."""
        row = self._session.get(MemoryModel, str(memory_id))
        return _row_to_memory(row) if row else None

    def update(self, memory: Memory) -> Memory:
        """Update an existing memory (must already exist)."""
        row = self._session.get(MemoryModel, str(memory.id))
        if row is None:
            raise KeyError(f"Memory not found: {memory.id}")

        row.content = memory.content
        row.agent = memory.agent
        row.timestamp = memory.timestamp.astimezone(UTC)
        row.confidence = memory.confidence
        row.importance = memory.importance
        row.scope = memory.scope.value
        row.task_id = memory.task_id
        row.memory_type = memory.memory_type.value
        row.parent_memory = str(memory.parent_memory) if memory.parent_memory else None
        row.supersedes = str(memory.supersedes) if memory.supersedes else None
        row.archived = memory.archived
        self._session.flush()
        logger.debug("Updated memory %s", memory.id)
        return _row_to_memory(row)

    def delete(self, memory_id: UUID) -> bool:
        """Permanently delete a memory."""
        row = self._session.get(MemoryModel, str(memory_id))
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        logger.debug("Deleted memory %s", memory_id)
        return True

    def archive(self, memory_id: UUID) -> Memory | None:
        """Soft-delete by marking archived=True."""
        row = self._session.get(MemoryModel, str(memory_id))
        if row is None:
            return None
        row.archived = True
        row.timestamp = utc_now()
        self._session.flush()
        return _row_to_memory(row)

    def list_local(
        self,
        agent: str,
        *,
        task_id: str | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        """List agent-private (local) memories."""
        return self._list(
            scope=MemoryScope.LOCAL,
            agent=agent,
            task_id=task_id,
            include_archived=include_archived,
            limit=limit,
        )

    def list_shared(
        self,
        *,
        task_id: str | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        """List collaborative (shared) memories."""
        return self._list(
            scope=MemoryScope.SHARED,
            agent=None,
            task_id=task_id,
            include_archived=include_archived,
            limit=limit,
        )

    def list_for_agent(
        self,
        agent: str,
        *,
        task_id: str | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        """
        List memories visible to an agent: their local memories plus shared memories.

        TODO: Apply trust-aware filtering at retrieval layer, not storage.
        """
        stmt = self._base_select(include_archived=include_archived)
        visibility = or_(
            and_(MemoryModel.scope == MemoryScope.LOCAL.value, MemoryModel.agent == agent),
            MemoryModel.scope == MemoryScope.SHARED.value,
        )
        stmt = stmt.where(visibility)
        if task_id is not None:
            stmt = stmt.where(MemoryModel.task_id == task_id)
        stmt = stmt.order_by(MemoryModel.timestamp.desc()).limit(limit)
        return [_row_to_memory(r) for r in self._session.scalars(stmt).all()]

    def list_by_task(
        self,
        task_id: str,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        """List all memories (local + shared) for a task."""
        stmt = (
            self._base_select(include_archived=include_archived)
            .where(MemoryModel.task_id == task_id)
            .order_by(MemoryModel.timestamp.desc())
            .limit(limit)
        )
        return [_row_to_memory(r) for r in self._session.scalars(stmt).all()]

    def supersede(self, old_id: UUID, new_memory: Memory) -> Memory:
        """
        Create a new memory that supersedes an existing one.

        The previous memory is archived; the new record links via supersedes.
        """
        existing = self.get(old_id)
        if existing is None:
            raise KeyError(f"Cannot supersede missing memory: {old_id}")

        new_memory = new_memory.model_copy(
            update={
                "supersedes": old_id,
                "parent_memory": new_memory.parent_memory or old_id,
            }
        )
        self.archive(old_id)
        return self.create(new_memory)

    def _list(
        self,
        *,
        scope: MemoryScope,
        agent: str | None,
        task_id: str | None,
        include_archived: bool,
        limit: int,
    ) -> list[Memory]:
        stmt = self._base_select(include_archived=include_archived).where(
            MemoryModel.scope == scope.value
        )
        if agent is not None:
            stmt = stmt.where(MemoryModel.agent == agent)
        if task_id is not None:
            stmt = stmt.where(MemoryModel.task_id == task_id)
        stmt = stmt.order_by(MemoryModel.timestamp.desc()).limit(limit)
        return [_row_to_memory(r) for r in self._session.scalars(stmt).all()]

    @staticmethod
    def _base_select(*, include_archived: bool) -> Select[tuple[MemoryModel]]:
        stmt = select(MemoryModel)
        if not include_archived:
            stmt = stmt.where(MemoryModel.archived.is_(False))
        return stmt
