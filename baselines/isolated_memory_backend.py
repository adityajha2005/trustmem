"""Collaborative Memory (2025) isolated baseline — no cross-user shared reads."""

from __future__ import annotations

import time
from typing import Any

from memory.schema import Memory
from retrieval.context import RetrievalContext
from retrieval.embedder import MemoryEmbedder
from retrieval.retriever import RetrievalResult, RetrievalStageTiming
from retrieval.scoring import ScoringWeights, rank_memories
from retrieval.vector_store import FaissVectorStore
from storage.repository import MemoryRepository
from utils.config import AppConfig


class IsolatedMemoryBackend:
    """
    Isolated-memory configuration from multi-user collaborative memory literature:
    each agent only retrieves memories they authored (no shared tier).
    """

    technique = "isolated_memory"

    def __init__(
        self,
        repository: MemoryRepository,
        config: AppConfig,
        *,
        faiss_path: str,
    ) -> None:
        self._repo = repository
        self._config = config
        self._embedder = MemoryEmbedder(
            config.embedding_model,
            offline=config.offline_retrieval,
        )
        self._store = FaissVectorStore(faiss_path, dimension=self._embedder.dimension)
        self._store.load()
        self._weights = ScoringWeights(semantic=1.0, trust=0.0, recency=0.0, importance=0.0)

    def index_memory(self, memory: Memory) -> None:
        if not memory.archived:
            vector = self._embedder.encode_documents([memory.content])[0]
            self._store.add(memory.id, vector)

    def sync_from_repository(self, *, task_id: str | None = None) -> int:
        if not task_id:
            return 0
        records = [
            m
            for m in self._repo.list_by_task(task_id, include_archived=False)
            if not m.archived
        ]
        if records:
            vectors = self._embedder.encode_documents([m.content for m in records])
            for memory, vec in zip(records, vectors, strict=True):
                self._store.add(memory.id, vec)
        return len(records)

    def retrieve(
        self,
        query: str,
        ctx: RetrievalContext,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        k = top_k or self._config.retrieval_top_k
        result = RetrievalResult(query=query)

        start = time.perf_counter()
        if not ctx.task_id:
            return result
        self.sync_from_repository(task_id=ctx.task_id)
        # Isolated: only memories authored by asker.
        candidates = [
            m
            for m in self._repo.list_by_task(ctx.task_id, include_archived=False)
            if m.agent == ctx.agent_id
        ]
        result.timings.append(
            RetrievalStageTiming("candidate_pool", (time.perf_counter() - start) * 1000)
        )
        if not candidates:
            return result

        start = time.perf_counter()
        allowed = {str(m.id) for m in candidates}
        query_vec = self._embedder.encode_query(query)
        scores = dict(self._store.search(query_vec, top_k=k * 2, allowed_ids=allowed))
        result.timings.append(
            RetrievalStageTiming("hybrid_search", (time.perf_counter() - start) * 1000)
        )

        ranked = rank_memories(
            candidates,
            scores,
            ctx,
            weights=self._weights,
            half_life_hours=self._config.recency_half_life_hours,
        )
        result.hits = [s.as_hit() for s in ranked[:k]]
        return result

    def close(self) -> None:
        self._store.save()

    def backend_info(self) -> dict[str, Any]:
        return {
            "implementation": "isolated_per_agent",
            "reference": "Collaborative Memory isolated tier (arXiv:2505.18279)",
        }
