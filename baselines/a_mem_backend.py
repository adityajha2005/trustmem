"""A-Mem inspired Zettelkasten-lite: semantic seed + linked neighbor expansion."""

from __future__ import annotations

import re
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


def _tags(content: str, *, max_tags: int = 8) -> set[str]:
    words = re.findall(r"[a-z]{4,}", content.lower())
    # Prefer informative tokens; take longest unique.
    unique = sorted(set(words), key=len, reverse=True)[:max_tags]
    return set(unique)


class AMemLinkedBackend:
    """
    Simplified A-Mem (arXiv:2502.12110): each note has tags; retrieval expands
    along tag overlap links after an initial semantic seed.
    """

    technique = "a_mem_linked"

    def __init__(
        self,
        repository: MemoryRepository,
        config: AppConfig,
        *,
        faiss_path: str,
        link_threshold: int = 1,
    ) -> None:
        self._repo = repository
        self._config = config
        self._link_threshold = link_threshold
        self._embedder = MemoryEmbedder(
            config.embedding_model,
            offline=config.offline_retrieval,
        )
        self._store = FaissVectorStore(faiss_path, dimension=self._embedder.dimension)
        self._store.load()
        self._tags_by_id: dict[str, set[str]] = {}
        self._weights = ScoringWeights(semantic=1.0, trust=0.0, recency=0.0, importance=0.0)

    def index_memory(self, memory: Memory) -> None:
        if memory.archived:
            self._tags_by_id.pop(str(memory.id), None)
            return
        vector = self._embedder.encode_documents([memory.content])[0]
        self._store.add(memory.id, vector)
        self._tags_by_id[str(memory.id)] = _tags(memory.content)

    def sync_from_repository(self, *, task_id: str | None = None) -> int:
        if not task_id:
            return 0
        records = [m for m in self._repo.list_by_task(task_id, include_archived=False)]
        self._tags_by_id.clear()
        if records:
            vectors = self._embedder.encode_documents([m.content for m in records])
            for memory, vec in zip(records, vectors, strict=True):
                self._store.add(memory.id, vec)
                self._tags_by_id[str(memory.id)] = _tags(memory.content)
        return len(records)

    def _linked_neighbors(self, seed_ids: list[str], pool: dict[str, Memory]) -> list[str]:
        neighbors: list[str] = []
        for sid in seed_ids:
            stags = self._tags_by_id.get(sid, set())
            if not stags:
                continue
            for mid, mem in pool.items():
                if mid == sid:
                    continue
                overlap = len(stags & self._tags_by_id.get(mid, set()))
                if overlap >= self._link_threshold:
                    neighbors.append(mid)
        return neighbors

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

        pool = {str(m.id): m for m in candidates}
        allowed = set(pool.keys())

        start = time.perf_counter()
        query_vec = self._embedder.encode_query(query)
        seed_hits = self._store.search(query_vec, top_k=max(3, k), allowed_ids=allowed)
        seed_ids = [mem_id for mem_id, _ in seed_hits]
        expanded_ids = list(dict.fromkeys(seed_ids + self._linked_neighbors(seed_ids, pool)))

        semantic_scores: dict[str, float] = {sid: 1.0 for sid in seed_ids}
        for nid in expanded_ids:
            semantic_scores.setdefault(nid, 0.6)

        result.timings.append(
            RetrievalStageTiming("hybrid_search", (time.perf_counter() - start) * 1000)
        )

        expanded_memories = [pool[mid] for mid in expanded_ids if mid in pool]
        ranked = rank_memories(
            expanded_memories,
            semantic_scores,
            ctx,
            weights=self._weights,
            half_life_hours=self._config.recency_half_life_hours,
        )
        result.hits = [s.as_hit() for s in ranked[:k]]
        for hit in result.hits:
            hit["linked"] = hit["memory_id"] not in seed_ids
        return result

    def close(self) -> None:
        self._store.save()

    def backend_info(self) -> dict[str, Any]:
        return {
            "implementation": "a_mem_zettelkasten_lite",
            "reference": "A-Mem agentic memory (NeurIPS 2025) tag-link expansion",
            "link_threshold": self._link_threshold,
        }
