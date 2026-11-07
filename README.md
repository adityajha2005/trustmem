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

*How often the top result was the outdated note, on 40 controlled tasks per model; the 95% interval on 0/40 is 0 to 9%. The agents' notes go in as plain shared memories with no `supersedes` links, so this measures trust-aware ranking alone. Writing the verifier's note first instead of last, so recency favors the stale notes, gives the same 0%.*

The cross-encoder mostly makes it worse. That's the part that surprised me.

The other half matters as much. In a second benchmark I made the most trusted agent wrong. With a confidently wrong verifier, trustmem put the right note first in **0 of 16** runs, while plain vector search got 9. trustmem ranks by who is trusted, so it is exactly as right as the agent it trusts. See [Benchmark B](#benchmark-b-30-harder-tasks) and [trust is a prior](#trust-is-a-prior).

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
| Correct, but verifier writes first (recency favors the stale note) | 0% |

It fails on exactly the tasks where trust is wrong. What you get is a different problem: instead of retrieval picking the wrong note, you have to decide who to trust, and that one you can measure and work on.

### Benchmark B: 30 harder tasks

The first benchmark always has one correction, made by the most trusted agent. Benchmark B (`experiments/hard_tasks.yaml`) breaks that on purpose: updates that never say they replace anything, agents that disagree, several versions of the same fact, and a verifier that is confidently wrong. The correct answer for every task is written in the task file before any model runs. Each cell shows how many tasks had the correct note on top, Grok 4.7 / GPT-6.1 Sol.

| Scenario | Tasks | Vector search | + cross-encoder | trustmem |
|---|---|---|---|---|
| Ambiguous update by the verifier | 4 | 3 / 1 | 1 / 1 | 4 / 4 |
| Ambiguous update by the same agent | 3 | 2 / 0 | 3 / 2 | 3 / 2 |
| Conflict, higher-trust agent is right | 4 | 2 / 1 | 3 / 0 | 4 / 4 |
| Conflict, lower-trust agent is right | 3 | 1 / 2 | 2 / 3 | 0 / 0 |
| Verifier is wrong, planner is right | 8 | 4 / 5 | 4 / 4 | 0 / 0 |
| Verifier corrects twice, newest is right | 4 | 1 / 1 | 2 / 1 | 1 / 2 |
| Later decision by a lower-trust agent | 4 | 1 / 0 | 0 / 0 | 0 / 0 |

When one agent clearly outranks the others and is right, trustmem gets it: 8 of 8 runs where the verifier updated a fact without saying it replaced anything, and 8 of 8 conflicts where the higher-trust agent was right. When the most trusted agent is wrong or out of date, it follows that agent every time, which is worse than plain search (0 of 16 wrong-verifier runs against 9 of 16). When the old and new notes come from the same agent, trust can't separate them and recency is too weak to help (one-week half-life, notes ten minutes apart): 5 of 6 for same-agent ambiguous updates, 3 of 8 for a verifier that corrected twice. The groups are small (3 to 8 tasks), so read these as patterns, not rates.

`python -m experiments.hard_run all --provider xai --provider openai` reruns it, and each run writes `label_check.csv` so you can confirm every generated note says the fact it was labelled with.

### Trust is a prior

The trust numbers in these runs come from roles: verifier 0.95, coder 0.70, planner 0.60. A role says who usually checks facts. It is not evidence that a given note is correct, and Benchmark B shows what happens when the two come apart. In practice you calibrate trust against things you can verify: tests that pass or fail, the production config, a human sign-off.

Future work, not built yet: trust that starts from the role and then moves with each verified outcome, tracked per agent and per topic. A verifier whose notes keep being contradicted by the source of truth would lose rank, and a coder who keeps being right would gain it. trustmem stores `confidence` on every memory, so the retriever doesn't change; what's missing is the loop that updates those numbers.

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
