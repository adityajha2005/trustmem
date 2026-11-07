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

*How often the top result was the outdated note. 40 tasks per model; the 95% interval on 0/40 is 0–9%.*

The cross-encoder mostly makes it worse. That's the part that surprised me.

## What trustmem does differently

Every memory carries who wrote it, who can see it, how confident the author was, and whether it replaced an older note. Retrieval uses that in two places:

- **Before search:** drop what the asking agent isn't allowed to see, drop archived notes, and keep only the latest version of anything that was superseded. This step costs under a millisecond.
- **After search:** blend the cross-encoder score with the author's confidence, so a reranker can't push a confirmed correction below a stale note just because the stale one uses the same words as the question.

## Where it breaks

The fix is only as good as your confidence numbers. I replayed the same 120 runs with the confidence values deliberately corrupted:

| Confidence signal | Outdated note on top |
|---|---|
| Correct | 0% |
| Random noise (±0.15) on every agent | 17% |
| A confidently wrong agent on 27.5% of tasks | 28% |
| No signal (every agent the same) | 78%, same as plain search |

So it fails on exactly the tasks where trust is wrong, and nowhere else. It doesn't remove the problem; it turns "retrieval picks the wrong note" into "decide who to trust", which is a problem you can actually work on.

Other honest limits: 40 tasks per model, written by me, with one clean correction each. Pinning the most trusted note to the top can occasionally pin the wrong note from that agent (it happened once in 120 runs). Below the top spot, an outdated note sometimes still ranks above the correction.

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

Details are in `docs/`: `ARCHITECTURE.md` for the pipeline, `BENCHMARKS.md` for every benchmark, and `ABLATION_FAILURES.md` for the failures that are supposed to happen.

## Layout

```
memory/       memory schema, scopes, supersession, contradictions
retrieval/    the retriever, trust scoring, reranker fusion
storage/      SQLite (WAL) storage
baselines/    the other memory designs used for comparison
benchmarks/   scripted scenarios
experiments/  agent experiment, ablations, plots
evaluation/   metrics
api/          FastAPI server
```

Built with SQLite, FAISS, sentence-transformers (BGE-small embeddings, MS MARCO MiniLM cross-encoder), FastAPI and pydantic.

## License

MIT
