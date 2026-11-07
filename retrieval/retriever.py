"""Collaborative memory retrieval pipeline."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from memory.contradictions import ContradictionRegistry
from memory.lineage import LineageIndex
from memory.schema import Memory, MemoryScope
from memory.scopes import can_read
from retrieval.ablation import PoolMode, RetrievalAblation
from retrieval.context import RetrievalContext
from retrieval.embedder import MemoryEmbedder
from retrieval.hybrid_search import fuse_pools, split_by_scope
from retrieval.rerank_fusion import RerankDiagnostics, fuse_rerank
from retrieval.reranker import CrossEncoderReranker
from retrieval.scoring import ScoredMemory, ScoringWeights, rank_memories
from retrieval.vector_store import FaissVectorStore
from storage.repository import MemoryRepository
from utils.config import AppConfig, load_config
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalStageTiming:
    name: str
    elapsed_ms: float


@dataclass
class RetrievalResult:
    query: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    timings: list[RetrievalStageTiming] = field(default_factory=list)
    filtered_archived: int = 0
    filtered_lineage: int = 0
    rerank_diagnostics: RerankDiagnostics | None = None
    # Set only when ctx.evidence is given: the top hit's claim has competing values that
    # neither external evidence nor a clear trust margin separates. Order is not changed.
    contested: bool = False


class CollaborativeMemoryRetriever:
    """
    Multi-stage collaborative retrieval:

      1. candidate_pool   — task / agent scoped SQL load
      2. scope_filter     — local vs shared visibility
      3. archive_filter   — exclude archived unless requested
      4. lineage_filter   — drop superseded / contradicted memories
      5. hybrid_search    — FAISS over local + shared pools
      6. score_rank       — semantic + trust + recency + importance
      7. rerank_fusion    — cross-encoder + trust-aware fusion (optional)
    """

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        config: AppConfig | None = None,
        embedder: MemoryEmbedder | None = None,
        vector_store: FaissVectorStore | None = None,
        reranker: CrossEncoderReranker | None = None,
        contradiction_registry: ContradictionRegistry | None = None,
        ablation: RetrievalAblation | None = None,
    ) -> None:
        self._repo = repository
        self._config = config or load_config()
        self._ablation = ablation or RetrievalAblation.full(
            use_reranker=self._config.use_reranker
        )
        self._contradictions = contradiction_registry or ContradictionRegistry()
        self._embedder = embedder or MemoryEmbedder(
            self._config.embedding_model,
            offline=self._config.offline_retrieval,
        )
        self._vector_store = vector_store or FaissVectorStore(
            self._config.faiss_index_path,
            dimension=self._embedder.dimension,
        )
        rerank_enabled = self._ablation.use_reranker
        self._reranker = reranker or CrossEncoderReranker(
            self._config.reranker_model,
            enabled=rerank_enabled,
            offline=self._config.offline_retrieval,
        )
        self._weights = self._ablation.scoring_weights(self._config)

    def index_memory(self, memory: Memory) -> None:
        """Embed and add a single memory to the FAISS index."""
        vector = self._embedder.encode_documents([memory.content])[0]
        self._vector_store.add(memory.id, vector)

    def index_memories(self, memories: list[Memory]) -> None:
        """Bulk-index memories (e.g. after load or rebuild)."""
        if not memories:
            return
        vectors = self._embedder.encode_documents([m.content for m in memories])
        for memory, vec in zip(memories, vectors, strict=True):
            self._vector_store.add(memory.id, vec)

    def save_index(self) -> None:
        self._vector_store.save()

    def retrieve(
        self,
        query: str,
        ctx: RetrievalContext,
        *,
        top_k: int | None = None,
        memory_ids: list[UUID] | None = None,
    ) -> RetrievalResult:
        k = top_k or self._config.retrieval_top_k
        candidate_k = self._config.retrieval_candidate_k
        result = RetrievalResult(query=query)

        candidates, t0 = self._timed("candidate_pool", self._load_candidates, ctx, memory_ids)
        result.timings.append(t0)

        candidates, t1 = self._timed("scope_filter", self._scope_filter, candidates, ctx)
        result.timings.append(t1)

        if self._ablation.agent_scoped_only and ctx.agent_id:
            candidates = [c for c in candidates if c.agent == ctx.agent_id]

        if self._ablation.enable_archive_filter:
            candidates, filtered_n, t2 = self._timed_archive(candidates, ctx)
        else:
            filtered_n = 0
            t2 = RetrievalStageTiming("archive_filter", 0.0)
        result.filtered_archived = filtered_n
        result.timings.append(t2)

        if self._ablation.enable_lineage_filter:
            candidates, lineage_n, t2b = self._timed_lineage(candidates, ctx)
        else:
            lineage_n = 0
            t2b = RetrievalStageTiming("lineage_filter", 0.0)
        result.filtered_lineage = lineage_n
        result.timings.append(t2b)

        if not candidates:
            return result

        search_stage = (
            "flat_search"
            if self._ablation.pool_mode == PoolMode.FLAT
            else "hybrid_search"
        )
        semantic_scores, t3 = self._timed(
            search_stage, self._semantic_search, query, candidates, ctx, candidate_k
        )
        result.timings.append(t3)

        ranked, t4 = self._timed(
            "score_rank",
            rank_memories,
            candidates,
            semantic_scores,
            ctx,
            weights=self._weights,
            half_life_hours=self._config.recency_half_life_hours,
        )
        result.timings.append(t4)

        hits = [s.as_hit() for s in ranked[:candidate_k]]
        if self._ablation.use_reranker and hits:
            hits, diag, t5 = self._timed_rerank_fusion(query, hits, k)
            result.timings.append(t5)
            result.rerank_diagnostics = diag
        else:
            hits = hits[:k]

        result.hits = hits
        if ctx.evidence is not None and hits:
            result.contested = _contested(ranked, hits[0]["memory_id"], ctx)
        logger.debug("Retrieved %d hits for query %r", len(hits), query)
        return result

    def _timed_rerank_fusion(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], RerankDiagnostics, RetrievalStageTiming]:
        """Run cross-encoder + trust-aware fusion + trust preservation."""
        start = time.perf_counter()
        fusion_cfg = self._ablation.fusion_config(self._config)
        rerank_scores = self._reranker.predict_scores(query, hits)
        if self._ablation.naive_rerank:
            # Backward-compatible naive sort (used by ``with_naive_rerank``).
            new_hits = self._reranker.rerank(query, hits, top_k=top_k)
            diag = RerankDiagnostics(
                rerank_applied=rerank_scores is not None,
                fusion_applied=False,
            )
            elapsed = (time.perf_counter() - start) * 1000
            return new_hits, diag, RetrievalStageTiming("rerank_fusion", elapsed)

        new_hits, diag = fuse_rerank(
            hits,
            rerank_scores,
            config=fusion_cfg,
            top_k=top_k,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return new_hits, diag, RetrievalStageTiming("rerank_fusion", elapsed)

    def _semantic_search(
        self,
        query: str,
        candidates: list[Memory],
        ctx: RetrievalContext,
        top_k: int,
    ) -> dict[str, float]:
        if self._ablation.pool_mode == PoolMode.FLAT:
            return _flat_keyword_scores(query, candidates)
        return self._hybrid_semantic_search(query, candidates, ctx, top_k)

    def _hybrid_semantic_search(
        self,
        query: str,
        candidates: list[Memory],
        ctx: RetrievalContext,
        top_k: int,
    ) -> dict[str, float]:
        """Run FAISS search on local and shared pools, then fuse scores."""
        mode = self._ablation.pool_mode
        if mode == PoolMode.LOCAL_ONLY:
            candidates = [m for m in candidates if m.scope == MemoryScope.LOCAL]
        elif mode == PoolMode.SHARED_ONLY:
            candidates = [m for m in candidates if m.scope == MemoryScope.SHARED]

        if not candidates:
            return {}

        self.index_memories(candidates)

        allowed = {str(m.id) for m in candidates}
        query_vec = self._embedder.encode_query(query)

        local, shared = split_by_scope(candidates)
        local_ids = {str(m.id) for m in local}
        shared_ids = {str(m.id) for m in shared}

        if mode == PoolMode.LOCAL_ONLY:
            return dict(self._vector_store.search(query_vec, top_k=top_k, allowed_ids=local_ids))
        if mode == PoolMode.SHARED_ONLY:
            return dict(self._vector_store.search(query_vec, top_k=top_k, allowed_ids=shared_ids))

        local_hits = self._vector_store.search(
            query_vec, top_k=top_k, allowed_ids=local_ids or allowed
        )
        shared_hits = self._vector_store.search(
            query_vec, top_k=top_k, allowed_ids=shared_ids or allowed
        )

        if not local:
            return {mem_id: score for mem_id, score in shared_hits}
        if not shared:
            return {mem_id: score for mem_id, score in local_hits}

        return fuse_pools(local_hits, shared_hits)

    def _load_candidates(
        self,
        ctx: RetrievalContext,
        memory_ids: list[UUID] | None,
    ) -> list[Memory]:
        if memory_ids:
            loaded = [self._repo.get(mid) for mid in memory_ids]
            return [m for m in loaded if m is not None]
        if ctx.task_id:
            return self._repo.list_by_task(ctx.task_id, include_archived=True)
        if ctx.agent_id:
            return self._repo.list_for_agent(
                ctx.agent_id, task_id=ctx.task_id, include_archived=True
            )
        return []

    @staticmethod
    def _scope_filter(records: list[Memory], ctx: RetrievalContext) -> list[Memory]:
        return [
            r
            for r in records
            if r.scope in ctx.allowed_scopes and can_read(r, ctx.agent_id)
        ]

    @staticmethod
    def _archive_filter(
        records: list[Memory], ctx: RetrievalContext
    ) -> tuple[list[Memory], int]:
        if ctx.include_archived:
            return records, 0
        kept = [r for r in records if not r.archived]
        return kept, len(records) - len(kept)

    def _timed_archive(
        self, records: list[Memory], ctx: RetrievalContext
    ) -> tuple[list[Memory], int, RetrievalStageTiming]:
        start = time.perf_counter()
        filtered, n = self._archive_filter(records, ctx)
        return filtered, n, RetrievalStageTiming("archive_filter", (time.perf_counter() - start) * 1000)

    def _lineage_filter(
        self, records: list[Memory], ctx: RetrievalContext
    ) -> tuple[list[Memory], int]:
        """Drop superseded memories and apply open-contradiction policy."""
        lineage = LineageIndex.from_memories(records)
        kept, dropped = self._contradictions.filter_for_retrieval(
            records,
            lineage,
            include_archived=ctx.include_archived,
        )
        if ctx.evidence is not None:
            kept, more = ctx.evidence.filter(kept, ctx.now or datetime.now(UTC))
            dropped += more
        return kept, dropped

    def _timed_lineage(
        self, records: list[Memory], ctx: RetrievalContext
    ) -> tuple[list[Memory], int, RetrievalStageTiming]:
        start = time.perf_counter()
        filtered, n = self._lineage_filter(records, ctx)
        return filtered, n, RetrievalStageTiming("lineage_filter", (time.perf_counter() - start) * 1000)

    @staticmethod
    def _timed(name: str, fn, *args, **kwargs):  # noqa: ANN001
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], int):
            return out[0], RetrievalStageTiming(name, (time.perf_counter() - start) * 1000)
        return out, RetrievalStageTiming(name, (time.perf_counter() - start) * 1000)


def _flat_keyword_scores(query: str, candidates: list[Memory]) -> dict[str, float]:
    """Deterministic bag-of-words overlap baseline (no embeddings)."""
    tokens = {t for t in query.lower().split() if t}
    scores: dict[str, float] = {}
    for memory in candidates:
        content_tokens = {t for t in memory.content.lower().split() if t}
        if not tokens:
            scores[str(memory.id)] = 0.0
            continue
        overlap = len(tokens & content_tokens) / len(tokens)
        scores[str(memory.id)] = overlap
    return scores


def _contested(ranked: list[ScoredMemory], top_id: str, ctx: RetrievalContext) -> bool:
    """Does the top hit's claim have competing values that nothing separates?"""
    top = next((s.memory for s in ranked if str(s.memory.id) == top_id), None)
    if top is None or not top.claim_key:
        return False
    verdict = ctx.evidence.verdict(top.claim_key, ctx.now)
    if verdict.contested:
        return True
    if verdict.value is not None:
        return False
    best: dict[str | None, float] = {}
    for s in ranked:
        if s.memory.claim_key == top.claim_key:
            best[s.memory.claim_value] = max(best.get(s.memory.claim_value, 0.0), s.trust_score)
    top_two = sorted(best.values(), reverse=True)[:2]
    return len(top_two) == 2 and top_two[0] - top_two[1] < ctx.contested_margin


def build_retriever(
    repository: MemoryRepository,
    config: AppConfig | None = None,
    *,
    ablation: RetrievalAblation | None = None,
    contradiction_registry: ContradictionRegistry | None = None,
) -> CollaborativeMemoryRetriever:
    """Factory: load persisted FAISS index when available."""
    cfg = config or load_config()
    embedder = MemoryEmbedder(cfg.embedding_model, offline=cfg.offline_retrieval)
    store = FaissVectorStore(cfg.faiss_index_path, dimension=embedder.dimension)
    store.load()
    return CollaborativeMemoryRetriever(
        repository,
        config=cfg,
        embedder=embedder,
        vector_store=store,
        ablation=ablation,
        contradiction_registry=contradiction_registry,
    )


# Backward-compatible alias
CollaborativeRetrievalPipeline = CollaborativeMemoryRetriever
