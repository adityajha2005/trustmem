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

## Run stress-v1 (2026-11-07)

- **Calls:** 155 live calls, $0.27 in total, against the $5 cap. Every reply parsed and used the right claim key; none hit the token limit; every call came back from the model that was requested.
- **Use `eval-v2`.** `eval-v1` counts gpt-6-luna tasks 005 and 006 twice.
  - Cause: a second `generate` process overlapped with the first for about 10 seconds. That produced 5 duplicate calls, which cost about $0.0005.
  - The raw traces keep the duplicates. `eval-v2` drops them at scoring time (`duplicate_records_dropped` in its manifest).
  - Generate now holds a lock on each model directory, so this can't happen again.
- **Real agents behaved like the idealised ones.** Every model's verifier was wrong on all 12 frozen-mirror tasks; none noticed the old sync date. Every lower agent shown the current file was right. As a result, the trust transitions are identical across the three models, so this run tests the trust mechanism under real-agent outputs, not variation between models.
