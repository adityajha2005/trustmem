"""MemGPT/Letta-inspired tiered memory: pinned core + archival vector search."""

from __future__ import annotations

import time
from typing import Any

from memory.schema import Memory
from memory.scopes import can_read
from retrieval.context import RetrievalContext
from retrieval.embedder import MemoryEmbedder
from retrieval.retriever import RetrievalResult, RetrievalStageTiming
from retrieval.scoring import ScoringWeights, rank_memories
from retrieval.vector_store import FaissVectorStore
from storage.repository import MemoryRepository
from utils.config import AppConfig


class MemGPTTieredBackend:
    """
    Letta/MemGPT pattern (simplified): recent ``core_slots`` always in context,
    plus semantic search over full archival pool.
    """

    technique = "memgpt_tiered"

    def __init__(
        self,
        repository: MemoryRepository,
        config: AppConfig,
        *,
        faiss_path: str,
        core_slots: int = 3,
    ) -> None:
        self._repo = repository
        self._config = config
        self._core_slots = core_slots
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
            return
        vector = self._embedder.encode_documents([memory.content])[0]
        self._store.add(memory.id, vector)
        self._indexed.add(str(memory.id))

    def sync_from_repository(self, *, task_id: str | None = None) -> int:
        if not task_id:
            return 0
        records = self._repo.list_by_task(task_id, include_archived=False)
        if records:
            vectors = self._embedder.encode_documents([m.content for m in records])
            for memory, vec in zip(records, vectors, strict=True):
                self._store.add(memory.id, vec)
                self._indexed.add(str(memory.id))
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
        candidates = self._repo.list_by_task(ctx.task_id, include_archived=False)
        candidates = [
            r
            for r in candidates
            if r.scope in ctx.allowed_scopes and can_read(r, ctx.agent_id)
        ]
        result.timings.append(
            RetrievalStageTiming("candidate_pool", (time.perf_counter() - start) * 1000)
        )
        if not candidates:
            return result

        # Core tier: most recent memories always eligible (working context).
        core = sorted(candidates, key=lambda m: m.timestamp, reverse=True)[
            : self._core_slots
        ]
        core_ids = {str(m.id) for m in core}

        start = time.perf_counter()
        allowed = {str(m.id) for m in candidates}
        query_vec = self._embedder.encode_query(query)
        archival_hits = dict(
            self._store.search(query_vec, top_k=k * 2, allowed_ids=allowed)
        )
        result.timings.append(
            RetrievalStageTiming("hybrid_search", (time.perf_counter() - start) * 1000)
        )

        # Boost core tier in merged scores.
        semantic_scores: dict[str, float] = dict(archival_hits)
        for mem in core:
            mid = str(mem.id)
            semantic_scores[mid] = max(semantic_scores.get(mid, 0.0), 1.0)

        ranked = rank_memories(
            candidates,
            semantic_scores,
            ctx,
            weights=self._weights,
            half_life_hours=self._config.recency_half_life_hours,
        )

        # Pin core at front (dedupe).
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for mem in core:
            mid = str(mem.id)
            if mid in seen:
                continue
            seen.add(mid)
            hits.append(
                {
                    "memory_id": mid,
                    "content": mem.content,
                    "agent": mem.agent,
                    "scope": mem.scope.value,
                    "score": 1.0,
                    "tier": "core",
                }
            )
        for scored in ranked:
            mid = str(scored.memory.id)
            if mid in seen:
                continue
            seen.add(mid)
            hit = scored.as_hit()
            hit["tier"] = "archival"
            hits.append(hit)
            if len(hits) >= k:
                break

        result.hits = hits[:k]
        return result

    def close(self) -> None:
        self._store.save()

    def backend_info(self) -> dict[str, Any]:
        return {
            "implementation": "memgpt_tiered",
            "reference": "Letta/MemGPT core + archival tiers (simplified)",
            "core_slots": self._core_slots,
        }
