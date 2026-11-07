# Research scope and validity

This document states what the codebase **actually evaluates** — so results are not oversold as “beating Mem0” without context.

## What is research-grade here

### Core contribution (collaborative memory)

The primary system is **multi-agent collaborative memory**, not single-user chat memory:

| Mechanism | Implementation | Validated by |
|-----------|----------------|--------------|
| Local vs shared scope | `memory/scopes.py`, `scope_filter` stage | `collaborative_recall_v1` (leak + recall) |
| Trust-aware ranking | `retrieval/scoring.py` + weights | `trust_calibration_v1` |
| Trust-aware rerank fusion | `retrieval/rerank_fusion.py` | `trust_calibration_v1`, ablations `with_naive_rerank`, `no_trust_preservation` |
| Lineage / supersession | `memory/lineage.py`, `MemoryEvolution` | `contradiction_resolution_v1`, `long_horizon_v1` |
| Contradiction registry | `memory/contradictions.py` | Same + suppression metrics |
| Reproducibility | Seeds, manifests, `git_sha`, CSV/JSON export | `evaluation/harness.py`, `experiments/export.py` |

Benchmarks are **deterministic and LLM-free**: fixed agent writes, fixed queries, threshold-based pass/fail. That is appropriate for **mechanism testing** and CI, not for end-user answer quality.

### Ablation study (component isolates)

`python -m experiments.runner` runs an **11 variant × 4 scenario** grid with:

- Documented **expected failures** (`experiments/expected_failures.py`, `docs/ABLATION_FAILURES.md`)
- Negative controls (`local_only`, `shared_only`, `no_trust_scoring`)
- Regression baselines (`with_naive_rerank`)

If `full` passes all four scenarios and the five control cells fail every seed, the toggles and metrics are **wired correctly** — that is real experimental design, not demo fluff.

### Tests

~86 pytest cases cover metrics, fusion math, storage, API, scenarios, and ablation registry. Pytest forces `MEMORY_OFFLINE_RETRIEVAL=1` so CI does not depend on Hugging Face.

## Technique comparison (Mem0 vs simple RAG vs collaborative)

`python scripts/run_experiments.py techniques` compares **three real backends**:

| Technique | Implementation | What it represents |
|-----------|----------------|-------------------|
| **collaborative** | Full pipeline in this repo | **System under evaluation** |
| **simple_rag** | FAISS + semantic scores only | Strong naive baseline (no trust/lineage/contradiction) |
| **mem0** | `mem0ai` SDK, Qdrant, `infer=False` | External user-scoped memory product pattern |

**Important — asymmetric comparison (by design):**

- **Mem0** scopes by `user_id` (= agent). It does **not** model cross-agent shared pool + trust fusion. Low scores on `collaborative_recall_v1` are **expected** and support the thesis (“single-user memory layers miss multi-agent coordination”).
- **Simple RAG** sees shared memories but ignores lineage, contradictions, and trust preservation. It may match collaborative on some metrics but fail trust calibration or stale suppression.
- **Collaborative** should win on scenarios that need scope + trust + evolution. Claiming “SOTA over Mem0” requires stating **which scenarios** and **which capabilities** — not a single aggregate number.

Mem0 benchmark mode uses `infer=False` (raw text storage). That is fair for retrieval apples-to-apples but **does not** exercise Mem0’s LLM fact-extraction pipeline.

## What is not claimed (yet)

- No large-scale human evaluation or LLM-as-judge on answer quality in the benchmark suite.
- No live multi-tenant production load tests.
- Graph-based contradiction reasoning and several `TODO`s in `memory/` are extension stubs.
- Technique sweeps on **Python 3.14 + macOS** may crash when loading multiple PyTorch stacks; use 3.12/3.13 and `MEMORY_OFFLINE_RETRIEVAL=1` for stable runs (see README).

## Suggested framing

1. **Problem:** Multi-agent tasks need shared, evolving, trust-weighted memory — not isolated per-user stores.
2. **Method:** Scoped writes + staged retrieval + fusion + contradiction/lineage filters.
3. **Evaluation:** Four collaborative scenarios + ablation grid + optional external baselines (simple RAG, Mem0 OSS).
4. **Results:** Report per-scenario metrics; treat five ablation failures as controls; report technique comparison with capability caveats above.

## Code correctness notes

- Scenario oracles use explicit `expected` thresholds — failures are informative, not flaky LLM outputs.
- `ranking_inversion_rate` and `trust_preservation_rate` are defined in `evaluation/metrics.py` with tests.
- Rerank fusion is pure functions with dedicated tests (`tests/test_rerank_fusion.py`).
- Known operational issue: multiple PyTorch loads (BGE + reranker + Mem0) on macOS; mitigated via `baselines/runtime.py` and shared Mem0 client per sweep.
