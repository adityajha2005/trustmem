"""Full collaborative memory pipeline (production retriever)."""

from __future__ import annotations

from typing import Any

from memory.contradictions import ContradictionRegistry
from memory.schema import Memory
from retrieval.ablation import RetrievalAblation
from retrieval.context import RetrievalContext
from retrieval.retriever import CollaborativeMemoryRetriever, RetrievalResult, build_retriever
from storage.repository import MemoryRepository
from utils.config import AppConfig


class CollaborativeBackend:
    """Real trust-aware collaborative retriever backed by SQLite + FAISS."""

    technique = "collaborative"

    def __init__(
        self,
        repository: MemoryRepository,
        config: AppConfig,
        *,
        ablation: RetrievalAblation | None = None,
        contradiction_registry: ContradictionRegistry | None = None,
    ) -> None:
        self._repo = repository
        self._config = config
        self._retriever = build_retriever(
            repository,
            config,
            ablation=ablation,
            contradiction_registry=contradiction_registry,
        )

    def index_memory(self, memory: Memory) -> None:
        if not memory.archived:
            self._retriever.index_memory(memory)

    def sync_from_repository(self, *, task_id: str | None = None) -> int:
        if not task_id:
            return 0
        records = self._repo.list_by_task(task_id, include_archived=False)
        if records:
            self._retriever.index_memories(records)
        return len(records)

    def retrieve(
        self,
        query: str,
        ctx: RetrievalContext,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        if ctx.task_id:
            self.sync_from_repository(task_id=ctx.task_id)
        return self._retriever.retrieve(query, ctx, top_k=top_k)

    def close(self) -> None:
        try:
            self._retriever.save_index()
        except OSError:
            pass

    def backend_info(self) -> dict[str, Any]:
        ablation = self._retriever._ablation  # noqa: SLF001 — benchmark introspection
        return {
            "implementation": "collaborative_pipeline",
            "ablation": ablation.to_dict(),
            "offline_retrieval": self._config.offline_retrieval,
            "use_reranker": self._config.use_reranker,
        }

    @property
    def retriever(self) -> CollaborativeMemoryRetriever:
        return self._retriever
