"""Trust-aware reranker fusion for collaborative retrieval.

Pure functions only — no models loaded here. The cross-encoder lives in
``retrieval/reranker.py``; this module composes raw rerank scores with
trust / semantic / importance signals and applies trust-preservation
strategies that prevent high-confidence memories from being demoted.

Pipeline contract:

    score_rank stage produces hits with these float fields:
      semantic_score, trust_score, recency_score, importance_score, final_score

    rerank_fusion stage:
      1. cross-encoder predicts a raw score per hit (or skipped)
      2. raw scores are min-max normalized to [0, 1]
      3. fused_score = w_s*semantic + w_t*trust + w_r*rerank_norm + w_i*importance
      4. optional trust preservation:
           - preserve_top_trusted_k: pin top-K hits by trust at the front
           - trust_bucket_reranking : rerank only within consecutive trust
             buckets defined by ``trust_bucket_threshold``
      5. attach per-hit diagnostics (pre/post rank, deltas, normalized scores)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RerankFusionWeights:
    """Linear weights for the rerank fusion stage. Should sum to ~1.0."""

    semantic: float = 0.30
    trust: float = 0.30
    reranker: float = 0.30
    importance: float = 0.10

    def normalized(self) -> "RerankFusionWeights":
        total = self.semantic + self.trust + self.reranker + self.importance
        if total <= 0:
            return self
        return RerankFusionWeights(
            semantic=self.semantic / total,
            trust=self.trust / total,
            reranker=self.reranker / total,
            importance=self.importance / total,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "trust": self.trust,
            "reranker": self.reranker,
            "importance": self.importance,
        }


@dataclass(frozen=True)
class RerankFusionConfig:
    """Runtime knobs for the fusion stage."""

    weights: RerankFusionWeights = field(default_factory=RerankFusionWeights)
    preserve_top_trusted_k: int = 0
    trust_bucket_reranking: bool = False
    trust_bucket_threshold: float = 0.05
    trust_threshold: float = 0.0  # hits below this trust are never preserved
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "preserve_top_trusted_k": self.preserve_top_trusted_k,
            "trust_bucket_reranking": self.trust_bucket_reranking,
            "trust_bucket_threshold": self.trust_bucket_threshold,
            "trust_threshold": self.trust_threshold,
            "enabled": self.enabled,
        }


@dataclass
class RerankDiagnostics:
    """Per-result diagnostics for a single fusion call."""

    top1_changed: bool = False
    mean_abs_rank_delta: float = 0.0
    max_abs_rank_delta: int = 0
    pairwise_swaps: int = 0
    preserved_count: int = 0
    bucket_count: int = 0
    rerank_applied: bool = False
    fusion_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "top1_changed": self.top1_changed,
            "mean_abs_rank_delta": round(self.mean_abs_rank_delta, 6),
            "max_abs_rank_delta": self.max_abs_rank_delta,
            "pairwise_swaps": self.pairwise_swaps,
            "preserved_count": self.preserved_count,
            "bucket_count": self.bucket_count,
            "rerank_applied": self.rerank_applied,
            "fusion_applied": self.fusion_applied,
        }


def normalize_rerank_scores(scores: list[float]) -> list[float]:
    """Min-max normalize raw cross-encoder logits into [0, 1]."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [0.5] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def trust_buckets(
    hits: list[dict[str, Any]],
    threshold: float,
) -> list[list[int]]:
    """
    Partition hit indices into consecutive trust buckets.

    Hits are first sorted by ``trust_score`` descending. A new bucket starts
    whenever the gap between adjacent trust values strictly exceeds
    ``threshold``. Within a bucket, the reranker is allowed to reorder freely;
    bucket order is never crossed. This preserves coarse trust calibration
    while still capturing fine-grained semantic relevance.
    """
    if not hits:
        return []
    if threshold < 0:
        threshold = 0.0

    order = sorted(
        range(len(hits)),
        key=lambda i: float(hits[i].get("trust_score", 0.0)),
        reverse=True,
    )
    buckets: list[list[int]] = [[order[0]]]
    for prev_idx, cur_idx in zip(order, order[1:]):
        prev_t = float(hits[prev_idx].get("trust_score", 0.0))
        cur_t = float(hits[cur_idx].get("trust_score", 0.0))
        if (prev_t - cur_t) > threshold:
            buckets.append([cur_idx])
        else:
            buckets[-1].append(cur_idx)
    return buckets


def _fused_value(
    hit: dict[str, Any],
    rerank_norm: float,
    weights: RerankFusionWeights,
) -> float:
    return (
        weights.semantic * float(hit.get("semantic_score", 0.0))
        + weights.trust * float(hit.get("trust_score", 0.0))
        + weights.reranker * rerank_norm
        + weights.importance * float(hit.get("importance_score", 0.0))
    )


def _select_preserved(
    hits: list[dict[str, Any]],
    config: RerankFusionConfig,
) -> set[int]:
    """Return indices pinned at the front by trust preservation."""
    k = max(0, int(config.preserve_top_trusted_k))
    if k == 0 or not hits:
        return set()
    eligible = [
        i
        for i, h in enumerate(hits)
        if float(h.get("trust_score", 0.0)) >= config.trust_threshold
    ]
    if not eligible:
        return set()
    eligible.sort(
        key=lambda i: (
            float(hits[i].get("trust_score", 0.0)),
            float(hits[i].get("final_score", 0.0)),
        ),
        reverse=True,
    )
    return set(eligible[:k])


def _pairwise_swaps(pre_order: list[int], post_order: list[int]) -> int:
    """Count how many ordered pairs swapped between pre and post rankings."""
    if len(pre_order) != len(post_order):
        return 0
    pre_pos = {idx: pos for pos, idx in enumerate(pre_order)}
    swaps = 0
    n = len(post_order)
    for i in range(n):
        for j in range(i + 1, n):
            if pre_pos[post_order[i]] > pre_pos[post_order[j]]:
                swaps += 1
    return swaps


def fuse_rerank(
    hits: list[dict[str, Any]],
    rerank_scores: list[float] | None,
    *,
    config: RerankFusionConfig,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], RerankDiagnostics]:
    """
    Apply trust-aware reranker fusion to ``hits``.

    Inputs:
      - ``hits``: list of hit dicts emitted by ``score_rank`` (each contains
        ``semantic_score``, ``trust_score``, ``importance_score``, ``final_score``).
        Their order is the pre-rerank ordering.
      - ``rerank_scores``: raw cross-encoder logits aligned with ``hits``.
        Pass ``None`` (or empty list) to skip reranking; fusion still runs
        with reranker_norm=0.5 for every hit (so weights still apply but the
        rerank signal is uninformative).
      - ``config``: fusion configuration.
      - ``top_k``: truncate the returned list to this length (None = keep all).

    Output:
      - reranked hit list with extra diagnostic fields per hit.
      - aggregate diagnostics (top-1 change, swap count, bucket count, ...).
    """
    diag = RerankDiagnostics(
        rerank_applied=bool(rerank_scores),
        fusion_applied=config.enabled,
    )

    if not hits:
        return [], diag

    pre_ids = [str(h.get("memory_id")) for h in hits]
    pre_pos = {mem_id: pos for pos, mem_id in enumerate(pre_ids)}

    if rerank_scores and len(rerank_scores) == len(hits):
        normalized = normalize_rerank_scores([float(s) for s in rerank_scores])
    else:
        normalized = [0.5] * len(hits)

    # Annotate each hit with rerank diagnostics first; fused score next.
    weights = config.weights.normalized() if config.enabled else config.weights
    annotated: list[dict[str, Any]] = []
    for idx, (hit, raw_score, rerank_norm) in enumerate(
        zip(hits, rerank_scores or [None] * len(hits), normalized, strict=False)
    ):
        entry = dict(hit)
        entry["pre_rerank_score"] = float(hit.get("final_score", 0.0))
        entry["pre_rerank_rank"] = idx
        entry["rerank_score"] = (
            None if raw_score is None else round(float(raw_score), 6)
        )
        entry["rerank_score_norm"] = round(float(rerank_norm), 6)
        if config.enabled:
            fused = _fused_value(entry, rerank_norm, weights)
        else:
            fused = entry["pre_rerank_score"]
        entry["fused_score"] = round(float(fused), 6)
        annotated.append(entry)

    # Decide ordering strategy:
    #   1. trust_bucket_reranking takes precedence when enabled.
    #   2. else preserve_top_trusted_k pins the top-K by trust.
    #   3. else: fully sort by fused_score.
    if config.enabled and config.trust_bucket_reranking:
        buckets = trust_buckets(annotated, config.trust_bucket_threshold)
        diag.bucket_count = len(buckets)
        ordered_indices: list[int] = []
        for bucket in buckets:
            bucket_sorted = sorted(
                bucket, key=lambda i: annotated[i]["fused_score"], reverse=True
            )
            ordered_indices.extend(bucket_sorted)
    else:
        preserved = (
            _select_preserved(annotated, config) if config.enabled else set()
        )
        diag.preserved_count = len(preserved)
        if preserved:
            preserved_ordered = sorted(
                preserved,
                key=lambda i: (
                    float(annotated[i].get("trust_score", 0.0)),
                    annotated[i]["fused_score"],
                ),
                reverse=True,
            )
            rest = [i for i in range(len(annotated)) if i not in preserved]
            rest.sort(key=lambda i: annotated[i]["fused_score"], reverse=True)
            ordered_indices = preserved_ordered + rest
        else:
            ordered_indices = sorted(
                range(len(annotated)),
                key=lambda i: annotated[i]["fused_score"],
                reverse=True,
            )
        diag.bucket_count = 1 if annotated else 0

    final: list[dict[str, Any]] = []
    rank_deltas: list[int] = []
    for new_pos, idx in enumerate(ordered_indices):
        entry = annotated[idx]
        old_pos = pre_pos[str(entry.get("memory_id"))]
        delta = new_pos - old_pos
        entry["post_rerank_rank"] = new_pos
        entry["rank_delta"] = delta
        entry["preserved_by_trust"] = bool(
            config.enabled
            and not config.trust_bucket_reranking
            and idx
            in (_select_preserved(annotated, config) if config.enabled else set())
        )
        rank_deltas.append(abs(delta))
        final.append(entry)

    if final:
        diag.top1_changed = (
            str(final[0].get("memory_id")) != pre_ids[0] if pre_ids else False
        )
        diag.mean_abs_rank_delta = sum(rank_deltas) / len(rank_deltas)
        diag.max_abs_rank_delta = max(rank_deltas)
        diag.pairwise_swaps = _pairwise_swaps(
            list(range(len(pre_ids))),
            [pre_pos[str(h.get("memory_id"))] for h in final],
        )

    if top_k is not None and top_k >= 0:
        final = final[:top_k]
    return final, diag
