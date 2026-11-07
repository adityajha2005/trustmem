# Benchmark D artifacts

Design and pre-registration: `docs/BENCHMARK_D.md`. No live results yet. Pilot and full runs are added only after review.

| Path | Origin | Contents |
|---|---|---|
| `tasks/pilot.jsonl`, `tasks/full.jsonl` | controlled_env (generated, no LLM) | Frozen tasks: timelines, views, truth at query time |
| `runs/<run-id>/prompts.jsonl` | llm_agent | Each prompt once, keyed by `prompt_sha256` |
| `runs/<run-id>/<model>/agent_calls.jsonl` | llm_agent | One line per API call: requested and returned model, params, prompt hash, usage, cost, latency, raw reply |
| `runs/<run-id>/<model>/writes.jsonl` | llm_agent_write | Parsed notes per agent step, with view, truth at write time and parse status |
| `runs/<run-id>/<model>/checks.jsonl` | harness_check | Deterministic verification events (pre-query and post-query) |
| `runs/<run-id>/<eval>/<model>/scores.jsonl`, `trust.jsonl` | replay | Rankings and labels per method and task; trust snapshot after each learning step |
| `runs/<run-id>/<eval>/summary.json`, `report.md`, `manifest.json` | replay | Aggregates with k/n and 95% intervals; input hashes |
| `runs/<run-id>/SHA256SUMS` | | Checksums of everything in the run |

Rescore offline (no API calls): `python -m experiments.d_run evaluate --run-id <run-id> --eval-name <new-name>`.
