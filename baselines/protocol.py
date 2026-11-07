"""Common interface for memory technique backends."""

from __future__ import annotations

from typing import Any, Protocol

from memory.schema import Memory
from retrieval.context import RetrievalContext
from retrieval.retriever import RetrievalResult


class MemoryBackend(Protocol):
    """Technique-specific memory store + retrieval (benchmarks and demos)."""

    technique: str

    def index_memory(self, memory: Memory) -> None:
        """Index or update a single memory in the technique store."""
        ...

    def sync_from_repository(self, *, task_id: str | None = None) -> int:
        """Refresh the technique index from the canonical SQL repository."""
        ...

    def retrieve(
        self,
        query: str,
        ctx: RetrievalContext,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        ...

    def close(self) -> None:
        """Release external resources (vector DB, clients)."""
        ...

    def backend_info(self) -> dict[str, Any]:
        """Metadata for manifests and traces."""
        ...
