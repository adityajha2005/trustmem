"""Unit tests for trust-aware reranker fusion."""

from __future__ import annotations

import pytest

from evaluation.metrics import rerank_disruption_rate, trust_preservation_rate
from retrieval.rerank_fusion import (
    RerankFusionConfig,
    RerankFusionWeights,
    fuse_rerank,
    normalize_rerank_scores,
    trust_buckets,
)


def _hit(mem_id: str, *, semantic: float, trust: float, importance: float = 0.0) -> dict:
    return {
        "memory_id": mem_id,
        "content": f"content-{mem_id}",
        "semantic_score": semantic,
        "trust_score": trust,
        "recency_score": 0.5,
        "importance_score": importance,
        "final_score": 0.5 * semantic + 0.5 * trust,
    }


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_normalize_rerank_scores_min_max() -> None:
    assert normalize_rerank_scores([]) == []
    assert normalize_rerank_scores([1.0]) == [0.5]
    out = normalize_rerank_scores([-1.0, 0.0, 1.0])
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_rerank_scores_constant_input() -> None:
    out = normalize_rerank_scores([0.7, 0.7, 0.7])
    assert out == [0.5, 0.5, 0.5]


def test_trust_buckets_split_on_gap() -> None:
    hits = [
        _hit("a", semantic=0.5, trust=0.95),
        _hit("b", semantic=0.5, trust=0.70),
        _hit("c", semantic=0.5, trust=0.68),
    ]
    buckets = trust_buckets(hits, threshold=0.05)
    # 0.95 → its own bucket; 0.70 and 0.68 share a bucket (gap <= 0.05).
    assert len(buckets) == 2
    assert sum(len(b) for b in buckets) == 3


def test_trust_buckets_one_bucket_when_threshold_high() -> None:
    hits = [
        _hit("a", semantic=0.5, trust=0.95),
        _hit("b", semantic=0.5, trust=0.30),
    ]
    buckets = trust_buckets(hits, threshold=1.0)
    assert len(buckets) == 1
    assert len(buckets[0]) == 2


# ---------------------------------------------------------------------------
# Trust preservation behavior (the headline fix)
# ---------------------------------------------------------------------------


def test_preserve_top_trusted_k_pins_high_trust_hit() -> None:
    """Reranker prefers low-trust hits, but preservation pins the verifier."""
    hits = [
        _hit("planner", semantic=0.9, trust=0.6),
        _hit("coder", semantic=0.9, trust=0.7),
        _hit("verifier", semantic=0.5, trust=0.95),
    ]
    rerank_scores = [10.0, 9.5, -3.0]  # cross-encoder dislikes verifier wording
    cfg = RerankFusionConfig(
        weights=RerankFusionWeights(0.30, 0.30, 0.30, 0.10),
        preserve_top_trusted_k=1,
    )
    out, diag = fuse_rerank(hits, rerank_scores, config=cfg)
    assert out[0]["memory_id"] == "verifier"
    assert out[0]["preserved_by_trust"] is True
    assert diag.preserved_count == 1


def test_no_preservation_lets_reranker_invert() -> None:
    """Without preservation/buckets the reranker can override trust."""
    hits = [
        _hit("planner", semantic=0.9, trust=0.6),
        _hit("coder", semantic=0.9, trust=0.7),
        _hit("verifier", semantic=0.5, trust=0.95),
    ]
    rerank_scores = [10.0, 9.5, -3.0]
    cfg = RerankFusionConfig(
        weights=RerankFusionWeights(0.10, 0.10, 0.70, 0.10),
        preserve_top_trusted_k=0,
    )
    out, diag = fuse_rerank(hits, rerank_scores, config=cfg)
    assert out[0]["memory_id"] != "verifier"
    assert diag.preserved_count == 0


def test_trust_bucket_reranking_does_not_cross_buckets() -> None:
    """Bucket reranking keeps verifier above coder/planner."""
    hits = [
        _hit("planner", semantic=0.9, trust=0.6),
        _hit("coder", semantic=0.9, trust=0.7),
        _hit("verifier", semantic=0.5, trust=0.95),
    ]
    rerank_scores = [10.0, 9.5, -3.0]
    cfg = RerankFusionConfig(
        weights=RerankFusionWeights(0.10, 0.10, 0.70, 0.10),
        preserve_top_trusted_k=0,
        trust_bucket_reranking=True,
        trust_bucket_threshold=0.05,
    )
    out, diag = fuse_rerank(hits, rerank_scores, config=cfg)
    assert out[0]["memory_id"] == "verifier"
    assert diag.bucket_count >= 2


# ---------------------------------------------------------------------------
# Diagnostics fields
# ---------------------------------------------------------------------------


def test_fusion_diagnostics_emit_per_hit_fields() -> None:
    hits = [
        _hit("a", semantic=0.4, trust=0.9),
        _hit("b", semantic=0.9, trust=0.4),
    ]
    out, diag = fuse_rerank(
        hits,
        [0.5, 0.9],
        config=RerankFusionConfig(),
    )
    for entry in out:
        assert "pre_rerank_rank" in entry
        assert "post_rerank_rank" in entry
        assert "rank_delta" in entry
        assert "rerank_score_norm" in entry
        assert "fused_score" in entry
    diag_dict = diag.to_dict()
    assert diag_dict["fusion_applied"] is True
    assert diag_dict["rerank_applied"] is True


def test_fusion_skipped_without_rerank_scores() -> None:
    hits = [
        _hit("a", semantic=0.4, trust=0.9),
        _hit("b", semantic=0.9, trust=0.4),
    ]
    out, diag = fuse_rerank(hits, None, config=RerankFusionConfig())
    assert diag.rerank_applied is False
    assert all(h["rerank_score"] is None for h in out)


# ---------------------------------------------------------------------------
# Standalone metrics
# ---------------------------------------------------------------------------


def test_trust_preservation_rate_full() -> None:
    trust = {"a": 0.9, "b": 0.5, "c": 0.1}
    assert trust_preservation_rate(["a"], trust, k=1) == 1.0


def test_trust_preservation_rate_lost() -> None:
    trust = {"a": 0.9, "b": 0.5, "c": 0.1}
    assert trust_preservation_rate(["c"], trust, k=1) == 0.0


def test_rerank_disruption_rate_unchanged() -> None:
    assert rerank_disruption_rate(["a", "b", "c"], ["a", "b", "c"]) == 0.0


def test_rerank_disruption_rate_swap_top() -> None:
    assert rerank_disruption_rate(["a", "b", "c"], ["b", "a", "c"], k=2) == 1.0
