# Benchmark D: real LLM agents, deterministic ground truth

Benchmark D is new work, separate from Benchmarks A, B and C. Those stay frozen (A and B in `results/historical/`, C in `results/benchmark_c/`, its 620 tasks unchanged). This file is the design and pre-registration. It is committed with the frozen tasks, prompts and config before the evaluation code. Results are added in later commits, after the pilot and full runs are approved.

## Question

Benchmark C used scripted notes, so its agents were right or wrong by construction. Benchmark D asks whether its findings hold when real LLM agents write the notes. Here an agent shown stale data may still answer correctly, may misread a file, may echo a teammate, or may write its note in a form the claim parser can't read. The aim is to find out whether trustmem works with real agents, not to show that it does. If the results contradict Benchmark C, that is reported as it stands.

## Three layers, kept apart

| Layer | What | Code | Artifacts |
|---|---|---|---|
| A. Controlled environment (generated, no LLM) | Small projects: `config/app.toml`, `docs/api_spec.md`, `requirements.lock`, `src/client.py`. Each task has a scripted timeline for one fact and fixes which document version each agent step sees | `experiments/d_env.py` | `results/benchmark_d/tasks/<split>.jsonl` (`origin: controlled_env`) |
| B. Real LLM agents | Planner, coder and verifier. One call per agent step. They see one document (or none) and the notes already written in the same task. They reply with `claim_key`, `claim_value`, optional `expires_at`, a free-text note and an unrelated note | `experiments/d_agents.py` | `runs/<id>/<model>/agent_calls.jsonl` (`origin: llm_agent`), `writes.jsonl` (`origin: llm_agent_write`) |
| C. Deterministic ground truth | The harness renders the environment file as it is at a given minute and reads the value back (`tomllib`, a regex, or by executing `src/client.py`) | `d_env.check`, `d_env.truth_at` | `runs/<id>/<model>/checks.jsonl` (`origin: harness_check`) |

### Who is right is never hardcoded

- A scenario controls only which document version each step is shown:
  - `current`: the file as it is at that moment;
  - `stale`: the previous version, as a handbook copy;
  - `decoy`: a design note with a value that was never true;
  - `stale_log`: an old CI log;
  - `check_output`: the latest harness check;
  - `none`: no document.
- Document titles never say whether they are current.
- Correctness is decided after the fact: the agent's normalised `claim_value` is compared with `truth_at(minute)`.
- An agent shown a stale view that answers correctly counts as correct. An agent shown the current file that misreads it counts as wrong.

### The verifier agent is not ground truth

- What the verifier writes is an ordinary agent note.
- Only harness checks create `VerificationEvent`s, with source kind `config` or `test`.
- Agent notes never become events.

### Claim parsing

Values are normalised deterministically: integers, versions (trailing `.0` dropped) and fixed option sets. Ambiguous text, such as "300 (was 1000)", gives `None`. Failures are counted, never dropped or repaired by an LLM.

The agent must choose the claim key from the topic's key list, which is the claim identity trustmem needs. Supplying the list is a limitation, but the agent still has to pick the right key and write the value and note in its own words.

## Scenarios

Ten scenarios, with tasks split evenly across them. Each task draws its topic, claim, values and view pattern from a seed. `CHANGE` means the environment moves to its next version; `CHECK` means a harness check runs before the query.

| Id | Topics | Steps | View patterns (one is drawn per task) |
|---|---|---|---|
| d01 stale config | config | planner, coder, verifier | stale/current/current; current/stale/current; stale/stale/current |
| d02 incorrect implementation | impl | planner, coder, verifier | decoy/current/current; decoy/decoy/current; current/decoy/current |
| d03 dependency update | deps | coder, CHANGE, planner, coder, verifier | current/stale/current/current; current/current/current/stale_log; current/stale/stale/current |
| d04 conflicting agents | api, config | planner, coder, verifier | current/stale/current; stale/current/stale; current/stale/stale; stale/current/current |
| d05 same-agent correction | api, impl | verifier, CHECK, verifier, planner, coder | stale_log/check_output/stale/current; stale_log/check_output/current/stale |
| d06 wrong verifier | config, impl | planner, coder, verifier | current/current/stale_log; current/current/decoy; current/stale/stale_log |
| d07 lower-trust agent correct | api, deps | planner, coder, verifier | current/stale/stale_log; current/stale/stale; current/decoy/stale |
| d08 expiring claims | config, api | planner, coder, verifier | override shown to all; base to the coder; base to the verifier |
| d09 repeated misinformation | api, impl | planner ×3, coder, verifier | stale×3/current/current; decoy×3/current/current; stale×3/none/current |
| d10 evidence-backed correction | deps, config | planner, coder, CHECK, verifier | stale/stale/check_output; stale/current/check_output; decoy/stale/check_output |

### Timing within a task

| Minute | What happens |
|---|---|
| 5 | Environment change, where the timeline has one |
| 10 × (step + 1) | Each step |
| 100 | A temporary override ends (d08) |
| 120 | Query |
| 130 | Post-query harness check |

Task *i* of a stream happens on day *i*. The 25 pilot tasks and the 100 full tasks come from different seeds (`pilot-v1`, `full-v1`), so inspecting the pilot reveals nothing about the full set. All models run the same tasks in the same order.

## Protocol (Option 1: one shared agent run per task)

1. **Generate (live API).** For each model, tasks run in stream order and each agent step makes one call.
   - What an agent sees depends only on the task script and the earlier notes of that task. It never depends on a retrieval method or on trust.
   - An unparseable reply gets one retry. If it still fails, the step is recorded with `parse_ok: false` and writes no note.
2. **Replay (offline).** Each of the four methods replays each model's recorded run with its own fresh SQLite and FAISS store and its own trust state. Per task:
   1. The task's notes are inserted. Each note gets the role prior as `confidence`, and its claim key, value and expiry.
   2. The query runs at minute 120, using trust as it is now. Learned methods use `BetaReliability.trust(agent, topic)`. `learned_evidence` also gets the pre-query harness checks as `Evidence`.
   3. Only after scoring is the post-query harness check passed to `learn()`.
   - Retrieval never updates trust. Agent-sourced events are ignored, inherited from `memory/reliability.py`.
   - `judge()` gives one outcome per agent per check, judged on its latest unexpired claim, so repeats count once and expired claims are not judged.
   - Pre-query checks feed evidence only. They are not also learned from, so an agent is never judged twice on the same claim.

Retrieval is scoped to the task, so notes never cross tasks or models. Only trust carries forward, and only within a model's stream.

### Methods

The four methods use exactly the Benchmark C conditions and parameters (n₀ = 4, λ = 0.90, m = 3, priors planner 0.60, coder 0.70, verifier 0.95).

| Method | Benchmark C condition |
|---|---|
| `dense` | plain vector search (`simple_rag`) |
| `static` | `full_rerank` with role priors |
| `learned` | `adaptive` (Beta per agent and topic) |
| `learned_evidence` | `adaptive_evidence` |

## Metrics

Every rate is reported as k/n with a Wilson 95% interval.

| Metric | Definition |
|---|---|
| `correct@1`, `correct@3` | The top-1 note, or any of the top 3, holds the true value at query time (on the task's claim key) |
| `stale@1` | The top note holds a value that was true earlier in the timeline |
| `refuted@1` | The top note holds a value that a pre-query harness check contradicted, among tasks with such a check |
| `off_key@1` | The top note is not about the queried fact |
| `oracle` | Some note holds the true value. This is the ceiling: no method can beat it |
| Conflict precision and recall | Recall: share of real conflicts (at least two distinct values among agents' latest unexpired claims) that are flagged `contested`. Precision: share of flagged tasks that are real conflicts. Defined only for `learned_evidence`; the other methods never flag |
| `correct@1 (lower-prior agent right)` | Among tasks where every correct agent has a lower prior than some wrong agent |
| Retrieval latency | Per method |
| Agent calls | Calls, prompt, completion, reasoning and cached tokens, cost and latency per model, from the call ledger |
| Agent behaviour | Parse failures, wrong keys, unreadable values, whether the note mentions the claimed value, expiry recorded, and correctness at write time per role and view |

Breakdowns are by model, method and scenario, and pooled. Per-scenario figures are descriptive: with about 10 tasks per scenario per model, intervals are roughly ±30 points.

## Pre-registered comparisons (full split only)

Each comparison is the paired difference in `correct@1`, with a 95% bootstrap interval that resamples tasks (2,000 resamples, seed 7), all models moving together. The pooled result is primary; per-model results are secondary.

| Comparison | Question |
|---|---|
| `learned` − `static` | Does learned trust beat static role trust when real agents write the notes? |
| `learned_evidence` − `learned` | Does evidence add anything on top of learned trust? |
| `learned_evidence` − `dense` | Does the full method beat plain vector search? |

Each comparison is supported if its interval lies above 0, contradicted if it lies below 0, and inconclusive otherwise. No direction is assumed. Benchmark C found both wins and losses, including dense beating every trust-aware variant when there was no verification. The pilot is exploratory and tests no hypothesis; it is for checking traces, failure modes, tokens and cost.

## Calls, cost and safeguards

- **Models:** `grok-4.7` (xAI), `gpt-6.1-sol` and `gpt-6-luna` (OpenAI), the ids used in Benchmark A. They are set per model in `d_config.yaml` and chosen with `--model`, not through `XAI_MODEL` or `OPENAI_MODEL`.
- **Settings:** temperature 0, at most 2,000 output tokens.
- **No silent substitution:** every call records the requested and returned model id, and the run stops on a mismatch. Only a dated snapshot suffix of the same id is accepted.
- **Budget:** a hard cap checked before every call against recorded spend plus the next call's worst case ($10 for the pilot, $30 for the full run, all models together).
- **Pricing:** live generation refuses to start until `price_verified` is true for the model, which requires checking the official pricing page. The prices in the config come from secondary sources.
- **Credentials:** keys come from the environment through the existing `.env.local` loader (`utils.config`). They are never printed or stored.
- **Secret scan:** evaluation scans every artifact for key patterns and for the actual key values, and refuses to write checksums if it finds one.

## Reproduce offline

```
python -m experiments.d_run tasks --split pilot          # regenerates the frozen file (tests check it is identical)
python -m experiments.d_run estimate --split pilot       # call and cost estimate from the real prompts, no API
python -m experiments.d_run generate --split pilot --model gpt-6-luna --mock --run-id mock-pilot
python -m experiments.d_run evaluate --run-id mock-pilot --split pilot
```

`evaluate` reruns only from recorded traces, so rescoring a real run never calls an API. Evaluation outputs go to a named directory (`--eval-name`, default `eval-v1`) that is never overwritten. A fix means a new eval name or a new run id, and the old artifacts are kept.

## Known limitations (stated before any results)

- Option 1: learned trust changes what is retrieved and scored, not what agents do next.
- The environments are small and generated, and the author designed the scenarios.
- The claim-key list is supplied to the agents.
- Harness checks are assumed correct.
- An agent whose fact later changed is judged against the new value unless it recorded an expiry (inherited from `judge`).
- Temperature 0 doesn't make runs repeatable. A repeat of about 20 tasks on one model will measure run-to-run variation.
- About 10 tasks per scenario per model is enough only for pooled conclusions.
