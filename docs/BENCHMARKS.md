# Benchmarks reference

Complete guide to every evaluation in this repository: deterministic mechanism benchmarks, ablation grids, external technique comparisons, and LLM end-to-end suites. All benchmarks are **reproducible** (fixed seeds, exported JSON manifests, optional `git_sha`).

For research claims and asymmetric baselines, see [RESEARCH_SCOPE.md](./RESEARCH_SCOPE.md). For literature-aligned backends, see [RELATED_WORK.md](./RELATED_WORK.md). For expected ablation failures, see [ABLATION_FAILURES.md](./ABLATION_FAILURES.md).

---

## Overview

| Tier | Name | Cells (typical) | Uses LLM? | Purpose |
|------|------|-----------------|-----------|---------|
| **1** | Collaborative scenarios | 5 runs | No | Validate scope, lineage, trust, contradiction |
| **2** | Ablation grid | 44 / seed | No | Isolate pipeline components; negative controls |
| **3** | Technique comparison | 7 × 4 = 28 / seed | No | Compare real memory backends on same scenarios |
| **4a** | LLM eval — basic | 3 × 2 = 6 | Yes (Grok) | Smoke cross-agent shared recall vs Mem0 |
| **4b** | LLM eval — complex | 7 × 10 = 70 | Yes (Grok) | Hard multi-agent storyline + judge |

```text
                    ┌─────────────────────────────────────┐
                    │     Tier 1: Scenario benchmarks      │
                    │  (5 scenarios, threshold oracles)    │
                    └─────────────────┬───────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Tier 2: Ablations │      │ Tier 3: Techniques│      │ Tier 4: LLM eval  │
│ 11 variants × 4   │      │ 7 backends × 4    │      │ retrieve → answer │
│ scenarios         │      │ scenarios         │      │ → judge           │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

---

## Quick commands

| Goal | Command |
|------|---------|
| List scenarios | `python -m benchmarks.runner --list` |
| Run all Tier-1 scenarios | `python -m benchmarks.runner --scenario all --seed 42` |
| One scenario | `python -m benchmarks.runner --scenario trust_calibration_v1 --seed 42` |
| Ablation sweep | `python -m experiments.runner --seed 42` |
| CI-fast ablation | `python -m experiments.runner --ci --no-plots` |
| Technique sweep | `python scripts/run_experiments.py techniques --seed 42` |
| LLM eval (complex, default) | `python scripts/run_experiments.py llm-eval --suite complex --seed 42` |
| LLM eval (basic) | `python scripts/run_experiments.py llm-eval --suite basic --seed 42` |
| Check config | `python scripts/run_experiments.py check-env` |

**Stable local runs (macOS):** set in `.env.local`:

```bash
MEMORY_OFFLINE_RETRIEVAL=1
MEMORY_USE_RERANKER=false
```

**Real-model embeddings:** `MEMORY_OFFLINE_RETRIEVAL=0` on Python 3.12/3.13. Models are cached per process via `retrieval/model_cache.py`.

---

## Tier 1: Deterministic scenario benchmarks

**Runner:** `benchmarks/runner.py`  
**Code:** `benchmarks/scenarios/*.py`  
**Harness:** `evaluation/harness.py` → writes `experiments/runs/<scenario>_seed<N>.json`

Each scenario:

1. Seeds a scripted multi-agent write sequence into isolated in-memory SQLite + FAISS.
2. Runs fixed queries through the collaborative retriever (or technique backend when used from Tier 3).
3. Computes metrics and compares to `expected` thresholds (`>=` floor or `("<=", ceiling)`).
4. Exports per-query traces (retrieved IDs, hits, stage timings).

### Scenarios

#### `session_recall_v1`

| Field | Value |
|-------|--------|
| **Intent** | Minimal smoke: Agent A writes shared fact; Agent B retrieves it |
| **Agents** | `agent_a`, `agent_b` |
| **Primary metrics** | `recall_at_k` (via `CollaborativeMetrics`) |
| **Use** | CI / quick regression |

#### `contradiction_resolution_v1`

| Field | Value |
|-------|--------|
| **Intent** | Auth migration JWT → OAuth with verifier finding and contradiction resolution |
| **Workflow** | planner (JWT plan) → coder (JWT impl) → verifier (logout vuln) → coder (OAuth+PKCE) → resolve |
| **Task ID** | `auth_migration` |
| **Pass thresholds** | `active_memory_accuracy` = 1.0, `stale_retrieval_rate` ≤ 0, `lineage_completeness` = 1.0, `contradiction_resolution_rate` = 1.0, `precision_at_1` = 1.0 |
| **Probes** | Stale JWT suppressed; OAuth active; lineage chain intact |

#### `long_horizon_v1`

| Field | Value |
|-------|--------|
| **Intent** | Four sessions of evolving feature requirements; retrieval must track latest active state |
| **Task ID** | `feature_multitenant` |
| **Pass thresholds** | `long_horizon_consistency` ≥ 0.65, `active_memory_accuracy` ≥ 0.65, `stale_retrieval_rate` ≤ 0.25, `mean_recall_at_3` ≥ 0.65 |
| **Probes** | Drift over time, supersession, low stale rate |

#### `collaborative_recall_v1`

| Field | Value |
|-------|--------|
| **Intent** | Agents A/B/C write shared + local memories; queries must respect scope |
| **Task ID** | `collab_recall` |
| **Pass thresholds** | `shared_recall_at_3` = 1.0, `scope_correctness` = 1.0, `no_private_leak` = 1.0, `mean_precision_at_3` ≥ 0.4 |
| **Probes** | Cross-agent shared recall; never return another agent’s `local` notes |

#### `trust_calibration_v1`

| Field | Value |
|-------|--------|
| **Intent** | Conflicting deployment claims; verifier has highest confidence; naive rerank must fail |
| **Task ID** | `trust_eval` |
| **Pass thresholds** | `precision_at_1` = 1.0, `contradiction_suppression_rate` = 1.0, `resolution_correct` = 1.0, `ranking_inversion_rate` ≤ 0 |
| **Probes** | Trust-aware fusion vs lexical rerank demoting verifier |

**Recorded (seed 42, collaborative `full`):** standalone manifests report `passed=true` for `trust_calibration_v1`, `contradiction_resolution_v1`, and `collaborative_recall_v1`.

### Manifest fields

| Field | Description |
|-------|-------------|
| `metrics` | Measured values |
| `expected` | Threshold spec per metric |
| `passed` | Boolean gate for CI |
| `queries` | Per-query retrieval trace |
| `seed`, `git_sha`, `started_at`, `config_snapshot` | Reproducibility envelope |

---

## Tier 2: Ablation grid

**Runner:** `experiments/runner.py` (also `python scripts/run_experiments.py ablation`)  
**Variants:** `retrieval/ablation.py` → `ABLATION_VARIANTS`  
**Scenarios:** 4 core scenarios (no `session_recall_v1`)

### Grid size

```text
11 variants × 4 scenarios = 44 cells per seed
```

### Variants

| Variant | What it disables / changes |
|---------|---------------------------|
| `full` | **Production baseline** — lineage, trust, archive, fusion (reranker from config) |
| `no_lineage_filter` | Lineage filter off |
| `no_trust_scoring` | Trust weight = 0 (mass → semantic) |
| `no_archive_filter` | Archived memories visible |
| `no_reranker` | Cross-encoder stage off |
| `with_naive_rerank` | Lexical rerank only (regression baseline) |
| `no_trust_preservation` | Fusion on, `preserve_top_trusted_k=0` |
| `trust_bucket_rerank` | Trust-bucketed reranking |
| `local_only` | Pool = local memories only |
| `shared_only` | Pool = shared memories only |
| `flat_baseline` | Flat pool ablation preset |
| `simple_rag` | Preset aligned with simple RAG backend |
| `mem0` | Preset aligned with Mem0 (`agent_scoped_only`) |

### Expected failures (5 cells — negative controls)

These **must fail** on every healthy seed. Do not “fix” retrieval to pass them.

| Variant | Scenario | Why it fails |
|---------|----------|--------------|
| `no_trust_scoring` | `trust_calibration_v1` | No trust signal → ranking inversions |
| `with_naive_rerank` | `trust_calibration_v1` | Lexical rerank demotes verifier |
| `no_trust_preservation` | `trust_calibration_v1` | Fusion without pinning high-trust |
| `local_only` | `collaborative_recall_v1` | Cannot see shared pool |
| `shared_only` | `collaborative_recall_v1` | Cannot see asker’s local notes |

Registry: `experiments/expected_failures.py`

### Recorded results (seed 42)

| Metric | Value |
|--------|--------|
| **Total passed** | **39 / 44** |
| **`full` variant** | **4 / 4** scenarios |
| Failed variants | `no_trust_scoring`, `with_naive_rerank`, `no_trust_preservation`, `local_only`, `shared_only` (each 3/4) |

### Artifacts

Under `experiments/runs/ablations/ablation_seed<N>/`:

| File | Contents |
|------|----------|
| `*_summary.json` | All cells with metrics + pass bit |
| `*_runs.csv` | Flat per-cell rows |
| `*_comparison.csv` | Mean metrics per variant + Δ vs `full` |
| `*_deltas.json` | Per-variant metric deltas |
| `*_comparison.md` / `*_report.tex` | Results tables |
| `*_latency.csv` | Mean stage latencies (ms) |

Plots: `experiments/plots/ablation_seed<N>/`

**Multiseed:** `python scripts/run_experiments.py multiseed --seeds 42,43,44`

---

## Tier 3: Technique comparison (retrieval-only)

**Runner:** `experiments/technique_benchmark.py`  
**CLI:** `python scripts/run_experiments.py techniques --seed 42`

Runs the **same four ablation scenarios** under each **real** memory backend (`baselines/`).

### Techniques (7)

| ID | Label | Implementation summary |
|----|-------|-------------------------|
| `collaborative` | Collaborative (trust-aware) | Full `CollaborativeMemoryRetriever` + `full` ablation |
| `simple_rag` | Simple RAG (FAISS) | Dense semantic search only |
| `mem0` | Mem0 OSS SDK | `mem0ai` + Qdrant, per-agent `user_id`, `infer=False` |
| `hybrid_bm25` | Hybrid BM25 + dense | BM25 + FAISS, RRF fusion |
| `memgpt_tiered` | MemGPT tiered | Pinned core slots + archival FAISS |
| `isolated_memory` | Isolated memory | Author-only retrieval (no shared reads) |
| `a_mem_linked` | A-Mem linked | Semantic seed + tag-neighbor expansion |

Mapping: `baselines/techniques.py`, factory: `baselines/factory.py`

### Grid size

```text
7 techniques × 4 scenarios = 28 cells per seed
```

### Recorded results (seed 42)

| Technique | Scenarios passed |
|-----------|------------------|
| collaborative | **4 / 4** |
| simple_rag | **3 / 4** |
| mem0 | **0 / 4** |
| hybrid_bm25, memgpt_tiered, isolated_memory, a_mem_linked | (see `techniques_seed42_summary.json`) |

Mem0 failing collaborative recall is **expected** (user-scoped memory, not shared task pool).

### Artifacts

| Path | Contents |
|------|----------|
| `experiments/runs/techniques/techniques_seed<N>/` | Summary JSON, CSV, deltas |
| `experiments/plots/techniques_seed<N>/` | Comparison figures |

---

## Tier 4: LLM end-to-end evaluation

**Runner:** `evaluation/llm_eval_runner.py`  
**CLI:** `python scripts/run_experiments.py llm-eval [--suite basic\|complex]`

Per cell:

1. Isolated in-memory DB + seed storyline.
2. Backend `retrieve(query)` → top-k context string.
3. **Grok** (`XAI_API_KEY`, default model from `XAI_MODEL`) answers from context only.
4. Same model **judges** answer against rubric (JSON score).

### Metrics

| Metric | Range | Definition |
|--------|-------|------------|
| **Retrieval term recall** | 0–1 | Fraction of `required_terms` appearing in retrieved context |
| **Answer term recall** | 0–1 | Fraction of `required_terms` in final answer |
| **Task success** | 0–1 | Required terms present and no `forbidden_terms` in answer |
| **Judge score** | 0–1 | LLM-as-judge rubric fit (`evaluation/llm_judge.py`) |
| **Forbidden in answer** | bool | Any forbidden term leaked |

### Suite: `basic` (`llm_eval_v1`)

**Tasks:** 2 · **Default techniques:** all 7 (or subset via `--technique`)

| Task ID | Asker | Stress |
|---------|-------|--------|
| `cross_agent_db_pool` | `agent_b` | Agent A’s shared DB fact |
| `team_infra_cache_queue` | `agent_b` | Redis (B) + RabbitMQ (C) shared facts |

**Seed:** `evaluation/llm_fixtures.py` → `seed_llm_eval_memories`

**Recorded (seed 42, 3 techniques × 2 tasks):**

| Technique | Mean judge |
|-----------|------------|
| collaborative | 1.00 |
| simple_rag | 1.00 |
| mem0 | 0.75 |

### Suite: `complex` (`llm_eval_complex_v1`)

**Tasks:** 10 · **Cells:** 7 × 10 = **70** per full sweep

**Seed:** `evaluation/llm_complex_suite.py` → `seed_complex_storyline`  
Single narrative: JWT→OAuth, deployment trust override, infra A–D, rollback supersede, local privacy notes.

| Task ID | Category | Required terms (summary) |
|---------|----------|---------------------------|
| `auth_current_after_migration` | Contradiction | oauth, pkce; forbid jwt |
| `deployment_verified_schedule` | Trust | wednesday, 09:00; forbid friday 14:00 |
| `multi_hop_capacity_planning` | Multi-hop | 50, rabbitmq, priority |
| `observability_slo` | Single-hop | 200, p99, latency |
| `rollback_active_procedure` | Lineage | canary, 5%; forbid v2.3.1 |
| `privacy_agent_a_local` | Scope | (empty); forbid verbose/debugger |
| `cross_agent_redis_ttl` | Cross-agent | redis, hour, ttl |
| `stale_jwt_suppression` | Stale | (empty); forbid hs256 |
| `infra_cache_and_db` | Multi-fact | redis, 50 |
| `contradiction_winner_rationale` | Contradiction | logout, oauth, pkce |

### Recorded results (complex, seed 42, Grok-3, full 70-cell run)

| Technique | Mean judge | Mean task success |
|-----------|------------|-------------------|
| hybrid_bm25 | **1.000** | 0.933 |
| collaborative | 0.950 | 0.933 |
| simple_rag | 0.950 | 0.933 |
| memgpt_tiered | 0.950 | 0.933 |
| a_mem_linked | 0.950 | 0.900 |
| isolated_memory | 0.700 | 0.200 |
| mem0 | 0.400 | 0.200 |

**Interpretation:**

- Recall-heavy tasks favor dense/hybrid RAG; collaborative ties simple RAG on judge.
- Mem0 / isolated memory low on **shared** tasks by design (scope model), not implementation bugs.
- `cross_agent_redis_ttl`: retrieval recall 1.0 but task success ≈ 0.33 on several backends — answer phrasing vs rubric (“hour”, “TTL”), not retrieval.

### Artifacts

| Path | Contents |
|------|----------|
| `experiments/runs/llm_eval/llm_eval_<suite>_seed<N>/` | `*_results.json`, `*_report.md` |
| `experiments/runs/llm_eval/llm_eval_complex_seed42_full.log` | Full-run console log (if saved) |
| `mem0_cache/` under run dir | Shared Mem0 client cache per sweep |

---

## Metrics glossary

Defined in `evaluation/metrics.py`.

| Metric | Meaning |
|--------|---------|
| `precision_at_k` | \|relevant ∩ top-k\| / k |
| `recall_at_k` | \|relevant ∩ top-k\| / \|relevant\| |
| `stale_retrieval_rate` | Fraction of hits that are archived/superseded |
| `active_memory_accuracy` | Fraction of hits in active lineage heads |
| `lineage_completeness` | Lineage chain coverage for evolution scenarios |
| `contradiction_resolution_rate` | Winner matches oracle after resolve |
| `ranking_inversion_rate` | High-confidence item ranked below low-confidence |
| `trust_preservation_rate` | Top-trusted memory kept after rerank fusion |
| `rerank_disruption_rate` | Top positions changed by reranker |
| `shared_recall_at_3` | Shared facts found in top-3 |
| `scope_correctness` | No illegal scope in results |
| `no_private_leak` | Other agents’ local memories absent |
| `long_horizon_consistency` | Mean active accuracy over session queries |
| `context_efficiency` | Useful slots / k |

---

## Artifact index (seed 42)

| Benchmark | Primary artifact |
|-----------|----------------|
| Scenario (each) | `experiments/runs/<scenario>_seed42.json` |
| Ablation | `experiments/runs/ablations/ablation_seed42/ablation_seed42_summary.json` |
| Techniques | `experiments/runs/techniques/techniques_seed42/techniques_seed42_summary.json` |
| LLM basic | `experiments/runs/llm_eval/llm_eval_seed42/llm_eval_seed42_results.json` |
| LLM complex | `experiments/runs/llm_eval/llm_eval_complex_seed42/llm_eval_complex_seed42_results.json` |

---

## How to read results (principal dev notes)

1. **Tier 1 + 2** prove mechanisms (scope, lineage, trust fusion). Trust Tier-1/2 over Tier-4 LLM judge for those claims.
2. **39/44 ablation** is healthy; **5 failures are controls**, not regressions.
3. **Do not** claim “beats Mem0” from Tier-4 alone — Mem0 is user-scoped; compare on documented capability classes.
4. **Hybrid BM25** topping LLM judge suggests merging sparse+dense into collaborative `hybrid_search` (see README / discussion).
5. Stratify LLM tasks into **governance** (auth, trust, privacy, stale) vs **recall** (infra, multi-hop) before drawing architecture conclusions.

---

## Related docs

- [ARCHITECTURE.md](./ARCHITECTURE.md) — pipeline stages
- [RERANK_FUSION.md](./RERANK_FUSION.md) — trust-aware fusion math
- [ABLATION_FAILURES.md](./ABLATION_FAILURES.md) — negative controls detail
- [RELATED_WORK.md](./RELATED_WORK.md) — baseline citations
- [RESEARCH_SCOPE.md](./RESEARCH_SCOPE.md) — valid claims
