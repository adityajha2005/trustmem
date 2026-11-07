# Adaptive-trust stress (secondary exploratory experiment)

This is a separate experiment from the primary Benchmark D evaluation (`../runs/pilot-v2/`). Its results are never pooled with it. It is not independently preregistered. The design was frozen before execution: see `docs/BENCHMARK_D_ADAPTIVE_TRUST.md`.

| Path | Contents |
|---|---|
| `tasks.jsonl` | The 25 frozen controlled tasks (seed `stress-v1`), the same for every model |
| `runs/<run-id>/<model>/agent_calls.jsonl` | Real LLM calls: usage, cost, the returned model and the raw reply |
| `runs/<run-id>/<model>/writes.jsonl` | The parsed agent notes (`origin: llm_agent_write`) |
| `runs/<run-id>/<model>/checks.jsonl` | Harness checks, the ground truth (`origin: harness_check`) |
| `runs/<run-id>/prompts.jsonl` | Every prompt sent, deduplicated by hash |
| `runs/<run-id>/<eval>/` | Offline replay: per-model `scores.jsonl` and `trust.jsonl`, plus `summary.json`, `manifest.json` and `report.md` |
| `runs/<run-id>/SHA256SUMS` | Checksums of every file in the run |

To rescore offline without any API calls, use a new `--eval-name`; existing evaluations are never overwritten:

```
python -m experiments.d_run evaluate --split stress --run-id <run-id> --eval-name eval-v2
```
