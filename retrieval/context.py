"""Retrieval context for a requesting agent in a collaborative session."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from memory.evidence import Evidence
from memory.schema import Memory, MemoryScope


class RetrievalContext(BaseModel):
    """Who is asking, under what trust policy, and which task scope applies."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    task_id: str | None = None
    allowed_scopes: list[MemoryScope] = Field(default_factory=lambda: list(MemoryScope))
    min_trust: float = Field(default=0.0, ge=0.0, le=1.0)
    include_archived: bool = False
    # Trust for a note's author. None = the note's stored confidence (static trust, as in
    # Benchmarks A and B). Pass e.g. ``lambda m: rel.trust(m.agent, topic_of(m.claim_key))``.
    trust: Callable[[Memory], float] | None = None
    # Verification evidence: enables claim groups, expiry, refutation and the contested flag.
    evidence: Evidence | None = None
    now: datetime | None = None  # query time for expiry and evidence; None = wall clock
    contested_margin: float = 0.05
