"""Pydantic models for persistent collaborative memory."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class MemoryScope(StrEnum):
    """Visibility: local (agent-private) or shared (collaborative)."""

    LOCAL = "local"
    SHARED = "shared"


class MemoryType(StrEnum):
    """High-level memory category for retrieval and consolidation."""

    FACT = "fact"
    OBSERVATION = "observation"
    PLAN = "plan"
    REFLECTION = "reflection"
    MESSAGE = "message"


class Memory(BaseModel):
    """
    A single persistent memory unit.

    Shared memories are visible across agents on the same task; local memories
    are private to the authoring agent.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., min_length=1)
    agent: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=utc_now)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    scope: MemoryScope = MemoryScope.SHARED
    task_id: str | None = None
    memory_type: MemoryType = MemoryType.FACT
    parent_memory: UUID | None = None
    supersedes: UUID | None = None
    archived: bool = False
    # Claim identity (optional): notes sharing a claim_key are versions of, or competitors
    # for, one fact. The topic used for per-topic trust is the prefix before the first dot.
    claim_key: str | None = None
    claim_value: str | None = None
    valid_until: datetime | None = None

    @field_validator("valid_until")
    @classmethod
    def ensure_valid_until_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @property
    def is_local(self) -> bool:
        return self.scope == MemoryScope.LOCAL

    @property
    def is_shared(self) -> bool:
        return self.scope == MemoryScope.SHARED

    def with_archived(self, archived: bool = True) -> Self:
        return self.model_copy(update={"archived": archived})

