"""Retrieval ablation presets for component-level experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from retrieval.rerank_fusion import RerankFusionConfig, RerankFusionWeights
from retrieval.scoring import ScoringWeights
from utils.config import AppConfig


class PoolMode(StrEnum):
    """Which memory pools participate in candidate ranking."""

    HYBRID = "hybrid"
    LOCAL_ONLY = "local_only"
    SHARED_ONLY = "shared_only"
    FLAT = "flat"


@dataclass(frozen=True)
class RetrievalAblation:
    """
    Toggle retrieval pipeline components for ablation studies.

    ``full`` is the production baseline; other presets disable one axis each
    (or switch pool / ranking mode) for results tables.
    """

    name: str = "full"
    enable_lineage_filter: bool = True
    enable_trust_scoring: bool = True
    enable_archive_filter: bool = True
    use_reranker: bool = False
    pool_mode: PoolMode = PoolMode.HYBRID

    # Trust-aware reranker fusion overrides. ``None`` = inherit from AppConfig.
    enable_rerank_fusion: bool = True
    preserve_top_trusted_k: int | None = None
    trust_bucket_reranking: bool | None = None
    naive_rerank: bool = False  # legacy: sort by raw rerank score only
    agent_scoped_only: bool = False  # mem0-style: only memories authored by asker

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pool_mode"] = self.pool_mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalAblation:
        pool = data.get("pool_mode", PoolMode.HYBRID)
        if isinstance(pool, str):
            pool = PoolMode(pool)
        return cls(
            name=str(data.get("name", "custom")),
            enable_lineage_filter=bool(data.get("enable_lineage_filter", True)),
            enable_trust_scoring=bool(data.get("enable_trust_scoring", True)),
            enable_archive_filter=bool(data.get("enable_archive_filter", True)),
            use_reranker=bool(data.get("use_reranker", False)),
            pool_mode=pool,
            enable_rerank_fusion=bool(data.get("enable_rerank_fusion", True)),
            preserve_top_trusted_k=data.get("preserve_top_trusted_k"),
            trust_bucket_reranking=data.get("trust_bucket_reranking"),
            naive_rerank=bool(data.get("naive_rerank", False)),
            agent_scoped_only=bool(data.get("agent_scoped_only", False)),
        )

    def scoring_weights(self, config: AppConfig) -> ScoringWeights:
        """Return scoring weights; zero trust when trust ablation is off."""
        if not self.enable_trust_scoring:
            # Reallocate trust mass to semantic for a fair single-signal ablation.
            trust = config.trust_weight
            return ScoringWeights(
                semantic=config.semantic_weight + trust,
                trust=0.0,
                recency=config.recency_weight,
                importance=config.importance_weight,
            )
        return ScoringWeights(
            semantic=config.semantic_weight,
            trust=config.trust_weight,
            recency=config.recency_weight,
            importance=config.importance_weight,
        )

    def fusion_config(self, config: AppConfig) -> RerankFusionConfig:
        """Build the RerankFusionConfig for this ablation, given AppConfig."""
        trust_weight = (
            0.0 if not self.enable_trust_scoring else config.fusion_trust_weight
        )
        # When trust scoring is off, push that mass into semantic so weights still sum.
        semantic_weight = (
            config.fusion_semantic_weight + (config.fusion_trust_weight - trust_weight)
        )
        weights = RerankFusionWeights(
            semantic=semantic_weight,
            trust=trust_weight,
            reranker=config.reranker_weight,
            importance=config.fusion_importance_weight,
        )
        preserve_k = (
            self.preserve_top_trusted_k
            if self.preserve_top_trusted_k is not None
            else config.preserve_top_trusted_k
        )
        bucket = (
            self.trust_bucket_reranking
            if self.trust_bucket_reranking is not None
            else config.trust_bucket_reranking
        )
        if not self.enable_trust_scoring:
            # No trust signal => preservation/bucketing make no sense.
            preserve_k = 0
            bucket = False
        return RerankFusionConfig(
            weights=weights,
            preserve_top_trusted_k=int(preserve_k),
            trust_bucket_reranking=bool(bucket),
            trust_bucket_threshold=config.trust_bucket_threshold,
            trust_threshold=config.trust_threshold,
            enabled=self.enable_rerank_fusion and not self.naive_rerank,
        )

    @classmethod
    def full(cls, *, use_reranker: bool = False) -> RetrievalAblation:
        return cls(name="full", use_reranker=use_reranker)

    @classmethod
    def no_lineage_filter(cls) -> RetrievalAblation:
        return cls(name="no_lineage_filter", enable_lineage_filter=False)

    @classmethod
    def no_trust_scoring(cls) -> RetrievalAblation:
        return cls(name="no_trust_scoring", enable_trust_scoring=False)

    @classmethod
    def no_archive_filter(cls) -> RetrievalAblation:
        return cls(name="no_archive_filter", enable_archive_filter=False)

    @classmethod
    def no_reranker(cls) -> RetrievalAblation:
        return cls(name="no_reranker", use_reranker=False)

    @classmethod
    def with_reranker(cls) -> RetrievalAblation:
        return cls(name="with_reranker", use_reranker=True)

    @classmethod
    def with_naive_rerank(cls) -> RetrievalAblation:
        """Old behavior: cross-encoder ordering, no fusion, no preservation."""
        return cls(
            name="with_naive_rerank",
            use_reranker=True,
            enable_rerank_fusion=False,
            naive_rerank=True,
            preserve_top_trusted_k=0,
            trust_bucket_reranking=False,
        )

    @classmethod
    def no_trust_preservation(cls) -> RetrievalAblation:
        """Fusion enabled, but no preservation/bucketing (isolates fusion alone)."""
        return cls(
            name="no_trust_preservation",
            use_reranker=True,
            enable_rerank_fusion=True,
            preserve_top_trusted_k=0,
            trust_bucket_reranking=False,
        )

    @classmethod
    def trust_bucket_rerank(cls) -> RetrievalAblation:
        """Bucket-based rerank: rerank within consecutive trust buckets only."""
        return cls(
            name="trust_bucket_rerank",
            use_reranker=True,
            enable_rerank_fusion=True,
            trust_bucket_reranking=True,
            preserve_top_trusted_k=0,
        )

    @classmethod
    def local_only(cls) -> RetrievalAblation:
        return cls(name="local_only", pool_mode=PoolMode.LOCAL_ONLY)

    @classmethod
    def shared_only(cls) -> RetrievalAblation:
        return cls(name="shared_only", pool_mode=PoolMode.SHARED_ONLY)

    @classmethod
    def flat_baseline(cls) -> RetrievalAblation:
        """Keyword-overlap ranking without FAISS hybrid fusion."""
        return cls(
            name="flat_baseline",
            pool_mode=PoolMode.FLAT,
            enable_trust_scoring=False,
        )

    @classmethod
    def simple_rag(cls) -> RetrievalAblation:
        """Plain semantic RAG: FAISS hybrid search, no trust/lineage/archive/rerank."""
        return cls(
            name="simple_rag",
            enable_lineage_filter=False,
            enable_trust_scoring=False,
            enable_archive_filter=False,
            use_reranker=False,
            pool_mode=PoolMode.HYBRID,
        )

    @classmethod
    def mem0_style(cls) -> RetrievalAblation:
        """
        Mem0-like session memory: per-agent store only, semantic search.

        Mirrors mem0 ``user_id`` scoping (no cross-agent shared pool).
        """
        return cls(
            name="mem0",
            enable_lineage_filter=False,
            enable_trust_scoring=False,
            enable_archive_filter=False,
            use_reranker=False,
            pool_mode=PoolMode.HYBRID,
            agent_scoped_only=True,
        )


# Registry of ablation variants (baseline first).
ABLATION_VARIANTS: dict[str, RetrievalAblation] = {
    "full": RetrievalAblation.full(),
    "no_lineage_filter": RetrievalAblation.no_lineage_filter(),
    "no_trust_scoring": RetrievalAblation.no_trust_scoring(),
    "no_archive_filter": RetrievalAblation.no_archive_filter(),
    "no_reranker": RetrievalAblation.no_reranker(),
    "with_naive_rerank": RetrievalAblation.with_naive_rerank(),
    "no_trust_preservation": RetrievalAblation.no_trust_preservation(),
    "trust_bucket_rerank": RetrievalAblation.trust_bucket_rerank(),
    "local_only": RetrievalAblation.local_only(),
    "shared_only": RetrievalAblation.shared_only(),
    "flat_baseline": RetrievalAblation.flat_baseline(),
    "simple_rag": RetrievalAblation.simple_rag(),
    "mem0": RetrievalAblation.mem0_style(),
}


def list_variants() -> list[str]:
    return list(ABLATION_VARIANTS.keys())


def get_variant(name: str) -> RetrievalAblation:
    key = name.strip()
    if key not in ABLATION_VARIANTS:
        raise KeyError(f"Unknown ablation variant: {name!r}. Choose from: {list_variants()}")
    if key == "full":
        from utils.config import load_config

        return RetrievalAblation.full(use_reranker=load_config().use_reranker)
    return ABLATION_VARIANTS[key]
