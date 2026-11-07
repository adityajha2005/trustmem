"""INMS-style hybrid retrieval: BM25 + dense semantic fusion (RRF)."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from baselines.bm25 import bm25_scores
from memory.schema import Memory
from memory.scopes import can_read
from retrieval.context import RetrievalContext
from retrieval.embedder import MemoryEmbedder
from retrieval.retriever import RetrievalResult, RetrievalStageTiming
from retrieval.scoring import ScoringWeights, rank_memories
from retrieval.vector_store import FaissVectorStore
from storage.repository import MemoryRepository
from utils.config import AppConfig


def _rrf_fuse(
    rankings: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal rank fusion across ordered id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, mem_id in enumerate(ranking):
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridBM25Backend:
    """
    Hybrid dense + sparse retrieval (inspired by INMS / production hybrid search).

    No trust, lineage, or contradiction filters — isolates hybrid recall.
    """

    technique = "hybrid_bm25"

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
        self._indexed: set[str] = set()
        self._weights = ScoringWeights(semantic=1.0, trust=0.0, recency=0.0, importance=0.0)

    def index_memory(self, memory: Memory) -> None:
        if memory.archived:
            mid = str(memory.id)
            if mid in self._indexed:
                self._store.remove(memory.id)
                self._indexed.discard(mid)
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
        for stale in self._indexed - active_ids:
            self._store.remove(UUID(stale))
            self._indexed.discard(stale)
        if active:
            vectors = self._embedder.encode_documents([m.content for m in active])
            for memory, vec in zip(active, vectors, strict=True):
                self._store.add(memory.id, vec)
                self._indexed.add(str(memory.id))
        return len(active)

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
        else:
            candidates = []
        result.timings.append(
            RetrievalStageTiming("candidate_pool", (time.perf_counter() - start) * 1000)
        )

        candidates = [
            r
            for r in candidates
            if r.scope in ctx.allowed_scopes and can_read(r, ctx.agent_id)
        ]
        if not candidates:
            return result

        start = time.perf_counter()
        allowed = {str(m.id) for m in candidates}
        id_to_mem = {str(m.id): m for m in candidates}

        query_vec = self._embedder.encode_query(query)
        dense_hits = self._store.search(
            query_vec, top_k=candidate_k, allowed_ids=allowed
        )
        dense_ranked = [mem_id for mem_id, _ in dense_hits]

        bm25 = bm25_scores(query, [m.content for m in candidates])
        sparse_ranked = [
            str(candidates[i].id)
            for i in sorted(range(len(candidates)), key=lambda j: bm25[j], reverse=True)
        ]

        fused = _rrf_fuse([dense_ranked, sparse_ranked])
        semantic_scores = {mid: fused[mid] for mid in fused}
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
            "implementation": "hybrid_bm25_rrf",
            "reference": "INMS-style interactive memory sharing (BM25 + dense RRF)",
            "embedding_model": self._config.embedding_model,
        }
