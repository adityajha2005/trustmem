"""Standalone semantic RAG: FAISS + embeddings, no trust/lineage/contradiction pipeline."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from memory.schema import Memory
from memory.scopes import can_read
from retrieval.context import RetrievalContext
from retrieval.embedder import MemoryEmbedder
from retrieval.retriever import RetrievalResult, RetrievalStageTiming
from retrieval.scoring import ScoringWeights, rank_memories
from retrieval.vector_store import FaissVectorStore
from storage.repository import MemoryRepository
from utils.config import AppConfig


class SimpleRAGBackend:
    """
    Real minimal RAG backend.

    - Canonical writes remain in SQLite (benchmark scenarios).
    - Retrieval: embed query → FAISS inner-product search → rank by semantic score only.
    - No reranker, trust fusion, lineage, or archive/contradiction filters.
    """

    technique = "simple_rag"

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
        self._store = FaissVectorStore(
            faiss_path,
            dimension=self._embedder.dimension,
        )
        self._store.load()
        self._indexed: set[str] = set()
        # Semantic-only weights (no trust signal).
        self._weights = ScoringWeights(
            semantic=1.0,
            trust=0.0,
            recency=0.0,
            importance=0.0,
        )

    def index_memory(self, memory: Memory) -> None:
        if memory.archived:
            mem_id = str(memory.id)
            if mem_id in self._indexed:
                self._store.remove(memory.id)
                self._indexed.discard(mem_id)
            return
        vector = self._embedder.encode_documents([memory.content])[0]
        self._store.add(memory.id, vector)
        self._indexed.add(str(memory.id))

    def sync_from_repository(self, *, task_id: str | None = None) -> int:
        if not task_id:
            return 0
        records = self._repo.list_by_task(task_id, include_archived=True)

        active = [m for m in records if not m.archived]
        active_ids = {str(m.id) for m in active}
        stale = self._indexed - active_ids
        for mem_id in stale:
            self._store.remove(UUID(mem_id))
            self._indexed.discard(mem_id)

        if active:
            vectors = self._embedder.encode_documents([m.content for m in active])
            for memory, vec in zip(active, vectors, strict=True):
                self._store.add(memory.id, vec)
                self._indexed.add(str(memory.id))
        return len(active)

    def _scope_filter(self, records: list[Memory], ctx: RetrievalContext) -> list[Memory]:
        return [
            r
            for r in records
            if r.scope in ctx.allowed_scopes and can_read(r, ctx.agent_id)
        ]

    def retrieve(
        self,
        query: str,
        ctx: RetrievalContext,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        k = top_k or self._config.retrieval_top_k
        candidate_k = self._config.retrieval_candidate_k
        result = RetrievalResult(query=query)

        start = time.perf_counter()
        if ctx.task_id:
            self.sync_from_repository(task_id=ctx.task_id)
            candidates = self._repo.list_by_task(ctx.task_id, include_archived=False)
        elif ctx.agent_id:
            candidates = self._repo.list_for_agent(
                ctx.agent_id, task_id=ctx.task_id, include_archived=False
            )
        else:
            candidates = []
        result.timings.append(
            RetrievalStageTiming("candidate_pool", (time.perf_counter() - start) * 1000)
        )

        start = time.perf_counter()
        candidates = self._scope_filter(candidates, ctx)
        result.timings.append(
            RetrievalStageTiming("scope_filter", (time.perf_counter() - start) * 1000)
        )

        if not candidates:
            return result

        start = time.perf_counter()
        allowed = {str(m.id) for m in candidates}
        query_vec = self._embedder.encode_query(query)
        semantic_scores = dict(
            self._store.search(query_vec, top_k=candidate_k, allowed_ids=allowed)
        )
        result.timings.append(
            RetrievalStageTiming("hybrid_search", (time.perf_counter() - start) * 1000)
        )

        start = time.perf_counter()
        ranked = rank_memories(
            candidates,
            semantic_scores,
            ctx,
            weights=self._weights,
            half_life_hours=self._config.recency_half_life_hours,
        )
        result.timings.append(
            RetrievalStageTiming("score_rank", (time.perf_counter() - start) * 1000)
        )

        result.hits = [s.as_hit() for s in ranked[:k]]
        return result

    def close(self) -> None:
        self._store.save()

    def backend_info(self) -> dict[str, Any]:
        return {
            "implementation": "simple_rag_faiss",
            "embedding_model": self._config.embedding_model,
            "offline_retrieval": self._config.offline_retrieval,
            "faiss_index_path": str(self._store.index_path),
        }
