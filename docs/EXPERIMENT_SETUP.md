# Experiment environment setup

## 1. Configure `.env.local`

```bash
cp .env.example .env.local
```

Edit `.env.local` (never commit — it is gitignored):

| Variable | Real-model run | CI / fast ablation |
|----------|-----------|-------------------|
| `MEMORY_OFFLINE_RETRIEVAL` | `0` | `1` |
| `MEMORY_USE_RERANKER` | `true` | `false` |
| `MEMORY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | same |
| `MEMORY_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | same |

Optional API keys (`OPENAI_API_KEY`, `GROQ_API_KEY`, etc.) belong **only** in `.env.local`, never in `.env.example` or git. They are not logged in experiment manifests.

## 2. Enable real embeddings

Set in `.env.local`:

```env
MEMORY_OFFLINE_RETRIEVAL=0
```

Verify:

```bash
python scripts/run_experiments.py check-env
# Should print: Real semantic retrieval is ENABLED.
```

First run downloads models via Hugging Face / sentence-transformers. Optional: set `HF_TOKEN` for gated models or faster downloads.

## 3. Experiment profile YAML

For experiment defaults (persistent DB path, reranker on):

```bash
export MEMORY_EXPERIMENT_CONFIG=experiment
```

Loads `configs/experiment.yaml` instead of `configs/default.yaml`. `.env.local` still overrides.

## 4. Run sweeps

```bash
pip install -e ".[viz]"   # plots
python scripts/run_experiments.py check-env
python scripts/run_experiments.py ablation --seed 42
python scripts/run_experiments.py multiseed --seeds 42,43,44
python scripts/run_experiments.py plots --run-dir experiments/runs/ablations/ablation_seed42
```

Artifacts:

- `experiments/runs/ablations/<run_id>/` — JSON, CSV, deltas
- `experiments/plots/<run_id>/` — PNG + PDF figures
- `experiments/reports/<run_id>_report.md` — summary + tables

## 5. Offline vs real retrieval

| Mode | `MEMORY_OFFLINE_RETRIEVAL` | Behavior |
|------|---------------------------|----------|
| Offline | `1` | MD5 hash vectors, no model download, deterministic |
| Semantic | `0` | BGE embeddings, FAISS IP search, optional cross-encoder rerank |

Toggle any time; call `reload_config()` or restart the process so settings reload.

Pytest forces offline mode via `conftest.py` regardless of `.env.local`.

## 6. Interpreting failures (expected negative controls)

A finished sweep typically reports **5 cells below thresholds** out of 44—that is **correct**. Those five cells are negative controls and mechanism isolates; do not “fix” them.

| Variant | Scenario | Why it fails |
|---------|----------|--------------|
| `no_trust_scoring` | `trust_calibration_v1` | Trust signal disabled |
| `with_naive_rerank` | `trust_calibration_v1` | Lexical rerank without preservation |
| `no_trust_preservation` | `trust_calibration_v1` | Fusion without `preserve_top_trusted_k` |
| `local_only` | `collaborative_recall_v1` | Shared pool excluded |
| `shared_only` | `collaborative_recall_v1` | Local pool excluded |

`full` should pass all four scenarios. Runner output: `INFO: 5 cells below scenario thresholds (5 expected negative controls, 0 unexpected)`.

Discussion: **[ABLATION_FAILURES.md](./ABLATION_FAILURES.md)**. Fusion: **[RERANK_FUSION.md](./RERANK_FUSION.md)**.

## 7. Reproducibility checklist

1. Fix `--seed` and record `git_sha` in manifest JSON.
2. Commit `configs/experiment.yaml` and document `.env.local` **keys** (not values).
3. Pin model IDs in `.env.local` / YAML.
4. Archive `experiments/runs/ablations/<run_id>/` and `experiments/plots/<run_id>/`.
5. For variance, use `multiseed` and report mean ± std from `multiseed_*.json`.
6. Cite failure analysis from `docs/ABLATION_FAILURES.md` when reporting pass rates.
