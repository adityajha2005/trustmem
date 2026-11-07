"""Agent protocol for memory read/write in collaborative workflows."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from memory.schema import Memory
from retrieval.context import RetrievalContext
from retrieval.pipeline import RetrievalResult


class AgentMemoryClient(Protocol):
    """Interface each agent implementation should satisfy."""

    @property
    def agent_id(self) -> str: ...

    def write(self, memory: Memory) -> Memory: ...

    def read(
        self,
        query: str,
        ctx: RetrievalContext | None = None,
        *,
        top_k: int = 5,
    ) -> RetrievalResult: ...

    def resolve_contradiction(self, claim_id: UUID, resolution: str) -> None: ...
