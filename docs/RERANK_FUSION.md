# Trust-aware reranker fusion

This note documents the reranker refinement introduced before the results freeze, replacing the previous `rerank` stage with `rerank_fusion`.

## Motivation

Seed-42 ablation results (`docs/ABLATION_FAILURES.md`) showed that the cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-orders candidates by query–text lexical similarity. On `trust_calibration_v1` this **inverts** the desired confidence ordering:

- planner (conf 0.6) and coder (conf 0.7) write claims that lexically match the query better than the verifier (conf 0.95) whose authoritative correction uses different wording.
- Naive rerank → verifier demoted → `precision_at_1` = 0, `ranking_inversion_rate` = 0.67.
- The semantic improvement comes at the cost of collaborative trust calibration.

The fix preserves both signals via a fused scoring stage with explicit trust-preservation strategies.

## Pipeline change

Old:

```
candidate_pool → scope_filter → archive_filter → lineage_filter → hybrid_search → score_rank → rerank
```

New:

```
candidate_pool → scope_filter → archive_filter → lineage_filter → hybrid_search → score_rank → rerank_fusion
```

`rerank_fusion` runs only when `use_reranker=True`. When the reranker is off the stage is skipped (no behavioral change vs the old pipeline for that case).

## Fusion formula

For each candidate hit emerging from `score_rank`:

```
rerank_norm = min_max_normalize(cross_encoder_logits)        # [0, 1]
fused_score = w_s · semantic_score
            + w_t · trust_score
            + w_r · rerank_norm
            + w_i · importance_score
```

All four weights are configurable. Defaults sum to 1.0:

| Weight | Default | Config field | Env override |
|--------|--------:|--------------|--------------|
| `semantic` | 0.30 | `fusion_semantic_weight` | `MEMORY_FUSION_SEMANTIC_WEIGHT` |
| `trust`    | 0.30 | `fusion_trust_weight`    | `MEMORY_FUSION_TRUST_WEIGHT` |
| `reranker` | 0.30 | `reranker_weight`        | `MEMORY_RERANKER_WEIGHT` |
| `importance` | 0.10 | `fusion_importance_weight` | `MEMORY_FUSION_IMPORTANCE_WEIGHT` |

Recency is intentionally **not** part of fusion; it already shapes pre-rerank candidate ordering inside `score_rank`. This keeps fusion to a small, interpretable surface.

## Trust preservation strategies

Two complementary mechanisms ensure high-confidence memories cannot be lexically demoted into oblivion:

### 1. `preserve_top_trusted_k` (default `1`)

Before sorting by `fused_score`, the top-K candidates by `trust_score` are pinned at the head of the result. Remaining candidates are sorted normally and appended.

- `K = 0` → pure fusion (no preservation).
- `K = 1` → verifier-style memories are guaranteed top-1 when their trust is highest. **This is the default and what fixes `trust_calibration_v1` × `full`.**
- `K = N` → keeps the top-N trust ordering frozen; the reranker only operates on the tail.

### 2. `trust_bucket_reranking` (default `false`)

When enabled, candidates are partitioned into **consecutive trust buckets**. Sorted by trust descending, a new bucket starts whenever the gap between adjacent trust values exceeds `trust_bucket_threshold` (default `0.05`). The reranker reorders **within** a bucket but never **across** buckets.

Example with confidences `0.95 / 0.70 / 0.68` and threshold `0.05`:

- Bucket 1: `{0.95}` (verifier) — pinned at top.
- Bucket 2: `{0.70, 0.68}` (coder, planner) — reranker decides their relative order.

`preserve_top_trusted_k` and `trust_bucket_reranking` are mutually exclusive (bucket mode wins when both are enabled).

`trust_threshold` (default `0.0`) excludes candidates below a minimum trust from being eligible for preservation — useful when you do not want to pin low-confidence memories simply because they are the highest-trust available.

## Why this addresses ranking inversion

The naive rerank failure has two causes:

1. **Single-signal ordering** — the cross-encoder produces a relevance score and ordering is `argsort(rerank_score)`, ignoring trust.
2. **No safety net** — even one bad rerank pair on a high-stakes verifier claim flips the top-1.

Fusion attacks (1) by giving the reranker a 30% weight while keeping 30% on trust and 30% on pre-rerank semantic similarity. Preservation / bucketing attack (2) directly: trust-driven invariants are enforced before fused scores are sorted.

For the trust calibration scenario:

| Variant | precision@1 | inversion_rate | trust_preservation | rerank_disruption |
|---------|------------:|---------------:|-------------------:|------------------:|
| `with_naive_rerank` (old `full`) | 0.0 | 0.67 | 0.0 | high |
| `no_trust_preservation` | depends on weights | depends | depends | high |
| `trust_bucket_rerank` | 1.0 | 0.0 | 1.0 | within-bucket only |
| `full` (default = preservation k=1) | 1.0 | 0.0 | 1.0 | tail-only |
| `no_reranker` | 1.0 | 0.0 | 1.0 | 0.0 |

(Numbers from synthetic unit tests in `tests/test_rerank_fusion.py`. Full real-embedding numbers will be filled after the next seed-42 sweep.)

## Diagnostics

Each retrieved hit now carries:

| Field | Meaning |
|-------|---------|
| `semantic_score`, `trust_score`, `recency_score`, `importance_score` | Pre-rerank score components |
| `pre_rerank_score`, `pre_rerank_rank` | `score_rank` final score and 0-based rank |
| `rerank_score`, `rerank_score_norm` | Raw cross-encoder logit and min-max normalized value |
| `fused_score`, `post_rerank_rank`, `rank_delta` | Output of fusion + final position |
| `preserved_by_trust` | True if pinned by `preserve_top_trusted_k` |

Each `RetrievalResult` exposes `rerank_diagnostics`:

| Field | Meaning |
|-------|---------|
| `top1_changed` | Did the head of the list change vs pre-rerank? |
| `pairwise_swaps` | # pairs whose relative order flipped |
| `mean_abs_rank_delta`, `max_abs_rank_delta` | Position-shift magnitude |
| `preserved_count` | # hits pinned by trust preservation |
| `bucket_count` | # trust buckets formed (1 = no bucketing) |
| `rerank_applied`, `fusion_applied` | Capability flags for the run |

These flow through to benchmark traces via `evaluation.ablation_metrics.attach_query_trace` and into `experiments/runs/.../*_summary.json`.

## New evaluation metrics

`evaluation/metrics.py`:

- `trust_preservation_rate(retrieved, trust_by_id, k)` — fraction of the top-k by trust that survive in the top-k retrieved.
- `rerank_disruption_rate(pre, post, k)` — fraction of pre-rerank top-k positions that changed.

Both are surfaced on `trust_calibration_v1` and aggregated in the canonical ablation table.

## Ablation presets

| Preset | Fusion | Preservation | Bucketing | Naive sort | Purpose |
|--------|:------:|:------------:|:---------:|:----------:|---------|
| `full` | yes | k=1 (default) | no | no | Production default |
| `no_reranker` | n/a | n/a | n/a | n/a | Disable rerank stage entirely |
| `with_naive_rerank` | no | k=0 | no | yes | Reproduce pre-fusion behavior |
| `no_trust_preservation` | yes | k=0 | no | no | Isolate fusion math without preservation |
| `trust_bucket_rerank` | yes | k=0 | yes | no | Bucket-based alternative to preservation |

## How this changes interpretation of trust experiments

Before:

> "The reranker improves semantic precision on long-horizon and contradiction scenarios but breaks trust calibration. Trade-off."

After:

> "Naive rerank breaks trust calibration; trust-aware fusion with `preserve_top_trusted_k = 1` preserves it. The `with_naive_rerank` and `full` rows quantify the cost of the naive design and the recovery from the fix. `trust_bucket_rerank` provides a softer, less invasive alternative that may be preferable when fine-grained semantic reranking matters more than strict pinning."

When reporting results, include:

- `with_naive_rerank` × trust_calibration as the **regression** baseline.
- `full` × trust_calibration as the **calibrated** baseline.
- `no_trust_preservation` and `trust_bucket_rerank` as **mechanism ablations** isolating the contribution of each preservation strategy.
- `trust_preservation_rate` and `rerank_disruption_rate` as the head-line metrics for this story.
