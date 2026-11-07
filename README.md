# trustmem

Shared memory for multi-agent LLM systems that doesn't hand you the outdated answer.

## The problem

Put a few agents on one memory store and this happens a lot:

1. The planner writes "public API limit is 1,000 requests/min".
2. The coder builds on it and writes something close to the same thing.
3. The verifier checks with infra, finds the limit is now 300, and writes that down, saying the old note is outdated.
4. Someone asks the memory "what's the rate limit?" and gets 1,000.

Nothing is broken. The embeddings work and the reranker works. The old note just matches the question's wording better, and similarity search has no idea who wrote what or which note replaced which.

I wanted numbers on how often this actually happens, so I ran 40 tasks like the one above with real LLM agents writing their own notes, on three models:

| Retrieval | Grok 4.7 | GPT-6 Luna | GPT-6.1 Sol |
|---|---|---|---|
| Vector search | 68% | 88% | 78% |
| Vector search + MS MARCO cross-encoder | 70% | 82% | 88% |
| trustmem | **0%** | **0%** | **0%** |

*How often the top result was the outdated note. 40 tasks per model; the 95% interval on 0/40 is 0 to 9%.*

The cross-encoder mostly makes it worse. That's the part that surprised me.

## What trustmem does differently

Every memory carries who wrote it, who can see it, how confident the author was, and whether it replaced an older note. Retrieval uses that twice. Before the vector search, it drops what the asking agent isn't allowed to see, drops archived notes, and keeps only the latest version of anything that was superseded. This costs under a millisecond. After the search, it blends the cross-encoder score with the author's confidence, so a stale note that shares the question's wording can't push a confirmed correction down the list.

## Architecture

```
  planner      coder      verifier          any agent: a LangGraph node, a script, an API client
     │           │            │
     └───── write(Memory) ────┘              read(query, agent, task)
                 │                                     │
                 ▼                                     ▼
   ┌───────────────────────────┐     ┌─────────────────────────────────────────────────┐
   │ storage/                  │     │ retrieval/   (every stage is timed)              │
   │   SQLite (WAL)            │────▶│ 1 candidate_pool  SQL load by task and agent     │
   │   one memories table      │     │ 2 scope_filter    LOCAL notes only for author    │
   │   FAISS index             │     │ 3 archive_filter  drop archived notes            │
   └───────────────────────────┘     │ 4 lineage_filter  drop superseded notes and      │
                 ▲                   │                   open contradictions            │
                 │                   │ 5 search          FAISS over BGE-small vectors   │
   ┌───────────────────────────┐     │ 6 score_rank      similarity, trust, recency,    │
   │ memory/                   │     │                   importance                     │
   │   schema, scopes,         │────▶│ 7 rerank_fusion   cross-encoder fused with trust │
   │   lineage, contradictions │     └────────────────────────┬────────────────────────┘
   └───────────────────────────┘                              ▼
                                               ranked hits + per-stage diagnostics
```

### The memory record

`memory/schema.py` defines one pydantic model, `Memory`:

| Field | Used for |
|---|---|
| `agent` | who wrote it; the trust signal comes from here |
| `scope` | `SHARED` is readable by every agent on the task, `LOCAL` only by its author |
| `confidence`, `importance` | both between 0 and 1, set by the writer |
| `timestamp` | recency decay |
| `supersedes`, `parent_memory` | the chain from an old note to the note that replaced it |
| `archived` | hidden from retrieval unless you ask for it |
| `task_id`, `memory_type` | candidate pool and filtering |

`memory/lineage.py` builds the supersession chains and answers "what is the latest version of this". `memory/contradictions.py` keeps a registry of notes that conflict and how each conflict was resolved. Everything is stored in a single SQLite table (WAL mode) with the vectors in FAISS next to it.

### Retrieval, stage by stage

Stages 1 to 4 are plain filters. They run before any embedding work, so a note the agent can't see or that was replaced never reaches the ranker. Any later stage that ranks well can't bring it back.

Stage 5 is ordinary dense search over BGE-small-en-v1.5 embeddings.

Stage 6 scores each hit as `0.45·similarity + 0.25·trust + 0.15·recency + 0.15·importance`, with recency on a one-week half-life. The weights live in `configs/default.yaml`.

Stage 7 runs the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, min-max normalizes its logits inside the batch, and fuses them:

```
fused = 0.30·similarity + 0.30·trust + 0.30·rerank + 0.10·importance
```

Fusion alone isn't enough when the reranker strongly prefers the stale note, so there are two guards. By default the most trusted hit is pinned to the top (`preserve_top_trusted_k: 1`). The alternative, `trust_bucket_reranking`, groups hits into trust tiers and lets the reranker reorder only inside a tier. Pinning has the failure listed below. Buckets avoid it but give up a little precision on long histories. `docs/RERANK_FUSION.md` has the details.

Each result carries diagnostics: rank before and after reranking, whether trust pinned it, and how many notes each filter dropped. The benchmarks read those, and so can you when a result looks wrong.

### Turning stages off

`retrieval/ablation.py` defines `RetrievalAblation`, a set of switches for each stage (`no_lineage_filter`, `no_trust_scoring`, `with_naive_rerank`, and others). Pass one to `build_retriever(..., ablation=...)`. The agent experiment uses these to compare plain search, search plus reranker, and the full pipeline on the same notes.

## Where it breaks

The fix is only as good as your confidence numbers. I replayed the same 120 runs with the confidence values deliberately corrupted:

| Confidence signal | Outdated note on top |
|---|---|
| Correct | 0% |
| Random noise (±0.15) on every agent | 17% |
| A confidently wrong agent on 27.5% of tasks | 28% |
| No signal (every agent the same) | 78%, same as plain search |

It fails on exactly the tasks where trust is wrong. What you get is a different problem: instead of retrieval picking the wrong note, you have to decide who to trust, and that one you can measure and work on.

Other limits: 40 tasks per model, written by me, with one clean correction each. Pinning the most trusted note to the top can occasionally pin the wrong note from that agent (it happened once in 120 runs). Below the top spot, an outdated note sometimes still ranks above the correction.

## Setup

Python 3.12 or 3.13.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
pytest
```

Tests run offline. They use hash embeddings instead of real models, so no GPU and no downloads.

## Running the agent experiment

Copy `.env.example` to `.env.local` and set `XAI_API_KEY`, `XAI_MODEL`, `OPENAI_API_KEY` and `OPENAI_MODEL`. Then:

```bash
python -m experiments.real_run all --provider xai --provider openai
```

That generates the agents' notes (the only step that calls an API, cached and resumable), scores four retrieval setups, and writes a CSV of sampled answers for checking by hand. Everything lands in `experiments/runs/real_run/<provider>/`.

```bash
python -m experiments.real_run evaluate --provider xai   # re-score from cache, free
python -m experiments.real_run noisy --provider xai      # the corrupted-trust replay
python -m experiments.real_run all --provider fake --offline   # no API at all
```

The tasks are in `experiments/real_run_tasks.yaml`. Add your own; each one is a context, the original fact, the correction, and the question.

## Other benchmarks

There are also scripted, LLM-free scenarios and an ablation grid. Some of the ablations are meant to fail: each one removes a feature its scenario depends on, so a pass would mean the benchmark is broken.

| What | Command |
|---|---|
| Scripted scenarios | `python -m benchmarks.runner` |
| Ablation grid | `python -m experiments.runner --seed 42` |
| Compare against other memory designs (RAG, Mem0, BM25 hybrid, MemGPT- and A-MEM-style) | `python scripts/run_experiments.py techniques --seed 42` |
| API server | `uvicorn api.main:app --reload` |

`docs/BENCHMARKS.md` lists every benchmark and `docs/ABLATION_FAILURES.md` explains the failures that are supposed to happen.

## Layout

```
memory/       memory schema, scopes, supersession, contradictions
retrieval/    the retriever, trust scoring, reranker fusion
storage/      SQLite (WAL) storage
baselines/    the other memory designs used for comparison
benchmarks/   scripted scenarios
experiments/  agent experiment, ablations, plots
agents/       agent base class
orchestration/ LangGraph coordination graph
evaluation/   metrics
api/          FastAPI server
```

Built with SQLite, FAISS, sentence-transformers (BGE-small embeddings, MS MARCO MiniLM cross-encoder), FastAPI and pydantic.

## License

MIT
