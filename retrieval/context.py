"""Retrieval context for a requesting agent in a collaborative session."""

from __future__ import annotations

from pydantic import BaseModel, Field

from memory.schema import MemoryScope


class RetrievalContext(BaseModel):
    """Who is asking, under what trust policy, and which task scope applies."""

    agent_id: str
    task_id: str | None = None
    allowed_scopes: list[MemoryScope] = Field(default_factory=lambda: list(MemoryScope))
    min_trust: float = Field(default=0.0, ge=0.0, le=1.0)
    include_archived: bool = False
    # TODO: Temporal reasoning — temporal_cutoff, valid_at for collaborative timelines
