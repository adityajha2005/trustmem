# Expected ablation failures (discussion)

**These failures are intentional.** They are negative controls and mechanism isolates—not bugs, not regressions, and **not** candidates for “fixes” that make every cell pass.

A healthy ablation sweep for this project reports **39 / 44 cells passed** per seed (seeds 42–44, fusion-enabled `full`). The five failing cells are **stable across seeds** and validate benchmark sensitivity.

Reference runs: `ablation_seed42`–`ablation_seed44`, `multiseed_42_44.json`, config `MEMORY_EXPERIMENT_CONFIG=experiment`.

---

## How to discuss these failures

Use this framing in the **Ablation study** or **Evaluation** subsection:

1. **State the grid:** 11 retrieval variants × 4 benchmark scenarios = 44 cells per seed; multiseed aggregation over seeds 42–44.
2. **Separate production from controls:** `full` is the system under evaluation; other variants disable one axis or reproduce a known failure mode.
3. **Celebrate expected failures:** A failed cell means the benchmark detected the intended violation (scope break, trust removal, naive rerank).
4. **Do not report “88% pass rate” without context**—either exclude negative controls from a headline pass rate or footnote the five expected failures.

**Suggested paragraph** (also in `experiments/expected_failures.py`):

> Of 44 (variant × scenario) cells per seed, five fail by design. Scope controls (`local_only`, `shared_only` on collaborative recall) confirm that pool restrictions are wired correctly. Trust ablations (`no_trust_scoring`) fail trust calibration when the trust signal is removed. Rerank ablations (`with_naive_rerank`, `no_trust_preservation`) fail the same scenario under lexical reranking without trust preservation, motivating our fusion stage. The production `full` variant passes all four scenarios including trust calibration when trust-aware fusion and `preserve_top_trusted_k=1` are enabled. We report raw pass/fail per cell and do not tune retrieval to satisfy negative controls.

---

## Expected failure registry (5 cells)

| Variant | Scenario | Category | Why it must fail | Role |
|---------|----------|----------|------------------|-------------------|
| `no_trust_scoring` | `trust_calibration_v1` | Trust ablation | Trust weight = 0; oracle is confidence order | Proves benchmark detects disabled trust scoring |
| `with_naive_rerank` | `trust_calibration_v1` | Rerank ablation | Lexical cross-encoder overrides confidence | **Regression baseline** before trust-aware fusion |
| `no_trust_preservation` | `trust_calibration_v1` | Rerank ablation | Fusion on, `preserve_top_trusted_k=0` | Shows preservation is required, not optional |
| `local_only` | `collaborative_recall_v1` | Scope control | SHARED pool excluded → shared recall &lt; 1 | Proves scope filter blocks collaboration |
| `shared_only` | `collaborative_recall_v1` | Scope control | LOCAL pool excluded → private recall fails | Complements `local_only` |

**Production baseline (`full`):** passes all four scenarios on seeds 42–44, including trust calibration (precision@1 = 1.0, ranking_inversion_rate = 0.0, trust_preservation_rate = 1.0).

---

## Per-failure detail

### 1. `no_trust_scoring` × `trust_calibration_v1`

**Mechanism:** `enable_trust_scoring=False` zeros trust weight and reallocates mass to semantic similarity (`retrieval/ablation.py`).

**Observed (seeds 42–44):** `ranking_inversion_rate` ≈ 0.33, `precision_at_1` = 0.

**Interpretation:** If this cell *passed*, the trust benchmark would be insensitive to removing trust—undermining the claim that collaborative retrieval is trust-aware.

---

### 2. `with_naive_rerank` × `trust_calibration_v1`

**Mechanism:** Reranker sorts by MS MARCO query–passage relevance only (`naive_rerank=True`, no fusion, no preservation). Planner/coder lines are lexically closer to “deployment time schedule” than the verifier’s authoritative correction.

**Observed (seeds 42–44):** `ranking_inversion_rate` ≈ 0.67, `precision_at_1` = 0, `trust_preservation_rate` = 0.

**Interpretation:** This is the **historical failure mode** that motivated trust-aware fusion. Cite it as the contrast to `full`, not as a broken system.

---

### 3. `no_trust_preservation` × `trust_calibration_v1`

**Mechanism:** Fusion runs with default weights but `preserve_top_trusted_k=0` and no trust bucketing. Reranker signal can still dominate trust in the fused score.

**Observed (seeds 42–44):** Same trust-calibration failure as naive rerank; `rerank_disruption_rate` = 1.0 (top positions reshuffled).

**Interpretation:** Isolates the **preservation mechanism** from the fusion formula—useful for an ablation figure with three bars: naive / fusion-only / fusion+preserve.

---

### 4. `local_only` × `collaborative_recall_v1`

**Mechanism:** `pool_mode=LOCAL_ONLY` drops all `MemoryScope.SHARED` candidates before search.

**Observed:** `shared_recall_at_3` &lt; 1.0 (recall ≈ 0.8 on seed 42).

**Interpretation:** Confirms agents cannot see shared task memory when the pool is restricted—expected for a scope negative control.

---

### 5. `shared_only` × `collaborative_recall_v1`

**Mechanism:** `pool_mode=SHARED_ONLY` drops LOCAL candidates; private-note queries miss.

**Observed:** recall ≈ 0.6 on seed 42.

**Interpretation:** Dual of `local_only`; together they bracket correct hybrid scope behavior that `full` satisfies.

---

## What passes (and why that matters)

| Scenario | What `full` demonstrates |
|----------|---------------------------|
| `contradiction_resolution_v1` | Lineage + archive + evolution suppress stale JWT |
| `long_horizon_v1` | Temporal consistency over evolving sessions |
| `collaborative_recall_v1` | Shared + local scope without cross-agent leaks |
| `trust_calibration_v1` | Confidence-calibrated ranking with fusion + preservation |

Variants that **pass** trust calibration on all seeds: `full`, `trust_bucket_rerank`, `no_reranker`, `no_lineage_filter`, `no_archive_filter`, `flat_baseline`, etc.—use them to show which components are orthogonal to trust calibration.

---

## Reporting checklist

| Do | Don’t |
|----|--------|
| Report per-cell pass/fail in appendix table | Hide failures or “fix” variants to reach 44/44 |
| Label five cells as **expected failure (negative control)** | Describe them as bugs or technical debt |
| Compare `full` vs `with_naive_rerank` on trust metrics | Claim ablations “beat” `full` on mean precision without scenario breakdown |
| Use multiseed mean ± std for `trust_preservation_rate`, `rerank_disruption_rate` | Use a single seed without noting variance |

**Headline pass rate (excluding negative controls):** 39 / 39 evaluative cells, or **39 / 44** with footnote “5 negative controls fail by design.”

---

## Runner output

The experiment CLI prints:

```text
INFO: 5 cells below scenario thresholds (5 expected negative controls, 0 unexpected)
```

An **unexpected** failure (variant not in the registry) warrants investigation. An **expected** failure warrants a sentence in the ablation discussion—not a code change.

---

## Related docs

- Fusion design: [RERANK_FUSION.md](./RERANK_FUSION.md)
- Experiment setup: [EXPERIMENT_SETUP.md](./EXPERIMENT_SETUP.md)
- Registry source: `experiments/expected_failures.py`

## Changelog

| Date | Run | Notes |
|------|-----|-------|
| 2026-05-23 | pre-fusion | 4 failures; `full`×trust failed (motivated fusion) |
| 2026-05-24 | `ablation_seed42–44` | 5 **expected** failures; `full`×trust passes all seeds |
| 2026-05-24 | docs | Reframed failures as negative controls |
