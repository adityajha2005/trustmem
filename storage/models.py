"""SQLAlchemy ORM models for persistent memory storage."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all storage tables."""


class MemoryModel(Base):
    """
    Persistent memory row.

    Supports local (agent-private) and shared (collaborative) scopes.
    """

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_agent_scope", "agent", "scope"),
        Index("ix_memories_task_scope", "task_id", "scope"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    agent: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    scope: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    task_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False, default="fact")
    parent_memory: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("memories.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    supersedes: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("memories.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
