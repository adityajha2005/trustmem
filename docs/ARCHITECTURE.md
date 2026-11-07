# Architecture: Collaborative Multi-Agent Memory

This project studies **how multiple agents coordinate persistent memory** — not single-user conversational recall. The design centers on scoped visibility, trust-aware retrieval, evolving shared state, and reproducible evaluation.

For validity of claims vs external baselines (Mem0, simple RAG), see [RESEARCH_SCOPE.md](./RESEARCH_SCOPE.md).

## Research focus

| Area | Design |
|------|--------|
| Primary unit | `Memory` with agent, scope, confidence, lineage fields |
| Visibility | `local` (agent-private) vs `shared` (collaborative) |
| Retrieval | Trust × semantic fusion, scope + archive gates |
| Evolution | `parent_memory` / `supersedes` links + contradiction registry |
| Orchestration | LangGraph multi-agent coordination graph |
| Evaluation | Cross-agent consistency, lineage, trust calibration |

## System map

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   agents/   │────▶│  orchestration/  │────▶│    memory/      │
│  (clients)  │     │  (LangGraph)     │     │ schema, lineage │
└──────┬──────┘     └──────────────────┘     └────────┬────────┘
       │                                                │
       ▼                                                ▼
┌─────────────┐                              ┌─────────────────┐
│ retrieval/  │◀─────────────────────────────│    storage/     │
│  pipeline   │                              │ SQLite + FAISS  │
└──────┬──────┘                              └─────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│ evaluation/ │◀────│   benchmarks/    │
│  + harness  │     │ scenarios (CI)   │
└─────────────┘     └──────────────────┘
```

## Collaborative retrieval pipeline

Modules: `embedder.py`, `vector_store.py`, `hybrid_search.py`, `scoring.py`, `reranker.py`, `retriever.py`.

Stages (timed for ablation):

1. **candidate_pool** — task / agent scoped SQL
2. **scope_filter** — local vs shared access
3. **archive_filter** — exclude archived by default
4. **hybrid_search** — FAISS over local + shared pools
5. **score_rank** — semantic + trust + recency + importance
6. **rerank** — optional cross-encoder (`use_reranker` in config)

## Persistence

- **Pydantic** domain model: `memory/schema.py` (`Memory`, scopes, types)
- **SQLAlchemy** ORM: `storage/models.py` (`MemoryModel`)
- **CRUD store**: `storage/sqlite_store.py` (`SQLiteMemoryStore`)
- **Init**: `storage/init_db.py` (WAL via `storage/database.py`)
- Single `memories` table with `parent_memory` / `supersedes` foreign keys
- **Local** (`scope=local`) vs **shared** (`scope=shared`) memory

## Benchmark-driven development

The evaluation layer is built for ablation-ready, reproducible research, not demo workflows.

### Architecture

- `benchmarks/_helpers.py` — `BaseScenario` with isolated `:memory:` SQLite + ephemeral FAISS
- `benchmarks/protocol.py` — public `CollaborativeScenario` / `BenchmarkScenario` surface
- `benchmarks/scenarios/*` — deterministic, LLM-free scenarios
- `benchmarks/runner.py` — CLI: `python -m benchmarks.runner --scenario all --seed 42`
- `evaluation/metrics.py` — pure metric primitives
- `evaluation/harness.py` — `BenchmarkReport` + `RunManifest`, seeds RNGs, persists JSON

Full benchmark catalog (commands, metrics, recorded results): [BENCHMARKS.md](./BENCHMARKS.md).

### Scenarios

| Scenario | Workflow | Primary metrics |
|----------|----------|-----------------|
| `contradiction_resolution_v1` | planner → coder (JWT) → verifier flag → coder (OAuth) | active_memory_accuracy, stale_retrieval_rate, lineage_completeness, contradiction_resolution_rate |
| `long_horizon_v1` | 4 sessions of evolving requirements + verification | long_horizon_consistency, mean_recall_at_3, stale_retrieval_rate, context_efficiency |
| `collaborative_recall_v1` | A/B/C agents, mixed local + shared scope | shared_recall_at_3, scope_correctness, no_private_leak |
| `trust_calibration_v1` | conflicting confidence levels, verifier override | precision_at_1, ranking_inversion_rate, contradiction_suppression_rate |
| `session_recall_v1` | minimal cross-agent recall smoke test | recall_at_k |

### Manifests

Each run writes `experiments/runs/<scenario>_seed<N>.json` containing:

- `metrics` — measured values (precision/recall, stale rate, etc.)
- `expected` — pass thresholds (`>=` floor or `("<=", x)` ceiling)
- `passed` — boolean gate for CI
- `queries` — per-query traces (retrieved ids, hits, filtered counts)
- `notes` — scenario-specific lineage / contradiction context
- `git_sha`, `seed`, `started_at`, `config_snapshot` — reproducibility envelope

## Ablation experiments

Component-level sweeps for results tables:

```bash
# Full sweep (8 variants × 4 scenarios)
python -m experiments.runner --seed 42

# CI-fast subset
python -m experiments.runner --ci

# Single variant
python -m experiments.runner --variant no_trust_scoring --scenario long_horizon_v1
```

Artifacts under `experiments/runs/ablations/<run_id>/`:

| File | Contents |
|------|----------|
| `*_summary.json` | All cells (variant × scenario) with canonical metrics |
| `*_runs.csv` | Flat per-cell rows |
| `*_comparison.csv` | Mean metrics per variant + Δ vs `full` |
| `*_deltas.json` | Per-variant metric deltas |
| `*_comparison.md` | Markdown table |
| `*_latency.csv` | Mean stage latencies (ms) |

Variants (`retrieval/ablation.py`):

- Pipeline ablations: `full`, `no_lineage_filter`, `no_trust_scoring`, `no_archive_filter`, `no_reranker`
- Reranker fusion ablations: `with_naive_rerank`, `no_trust_preservation`, `trust_bucket_rerank`
- Pool / search ablations: `local_only`, `shared_only`, `flat_baseline`

Toggles are passed via `RetrievalAblation` into `build_retriever(..., ablation=...)`.

### Trust-aware reranker fusion

The final retrieval stage is `rerank_fusion` (see `retrieval/rerank_fusion.py`):

```
fused_score = w_s · semantic + w_t · trust + w_r · rerank_norm + w_i · importance
```

Cross-encoder logits are min-max normalized within the candidate batch before fusion. Two trust-preservation strategies prevent high-confidence memories from being demoted by lexical reranking:

- `preserve_top_trusted_k` (default 1): pin top-K candidates by trust at the head, regardless of fused score.
- `trust_bucket_reranking`: partition candidates into consecutive trust buckets (gap > `trust_bucket_threshold` = new bucket); rerank only within each bucket.

Per-hit diagnostics (`pre_rerank_rank`, `post_rerank_rank`, `rank_delta`, `rerank_score_norm`, `fused_score`, `preserved_by_trust`) and result-level diagnostics (`top1_changed`, `pairwise_swaps`, `bucket_count`, …) are persisted in benchmark traces.

**Expected failures:** Five (variant × scenario) cells fail **by design** every seed—scope controls (`local_only`, `shared_only`), trust ablation (`no_trust_scoring`), rerank isolates (`with_naive_rerank`, `no_trust_preservation`). Production `full` passes all four scenarios. Registry: `experiments/expected_failures.py`. Discussion: `docs/ABLATION_FAILURES.md`. Fusion: `docs/RERANK_FUSION.md`.

## Visualization & reports

After each sweep, `export_experiment_with_artifacts()` generates:

| Output | Location |
|--------|----------|
| Figures (PNG + PDF) | `experiments/plots/<run_id>/` |
| Summary report (Markdown) | `experiments/reports/<run_id>_report.md` |
| LaTeX table snippet | `experiments/reports/<run_id>_report.tex` |

```bash
# Sweep + auto plots (default)
python -m experiments.runner --seed 42

# Regenerate plots from saved JSON
python -m experiments.visualize --run-dir experiments/runs/ablations/ablation_seed42

# Skip figures (CI / headless)
python -m experiments.runner --ci --no-plots
```

Plots: ablation comparison, contradiction rate, long-horizon consistency, precision/recall, stage latency (stacked), lineage suppression.

Dependencies: `pip install -e ".[viz]"` (matplotlib, pandas).

## Memory evolution

- `memory/lineage.py` — `LineageIndex`: supersession chains, `get_latest_memory`, `get_active_memories`
- `memory/contradictions.py` — `ContradictionRegistry`: register, `resolve_conflicts`, retrieval filtering
- `memory/temporal.py` — recency/trust tie-breaking, stale-by-age checks
- `memory/evolution.py` — `MemoryEvolution` facade (planner → coder → verifier workflows)

Retrieval applies a **lineage_filter** stage after archive filtering.

## Extension points (TODO)

- Graph-based contradiction reasoning
- Temporal memory graphs (`memory/temporal.py`)
- Adaptive trust evolution
- Probabilistic contradiction resolution
- Hybrid BM25 + vector retrieval
