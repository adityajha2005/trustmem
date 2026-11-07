# Benchmark D, secondary experiment: adaptive-trust stress

This is a **secondary exploratory experiment.** Its design was frozen and committed before any live call. It is **not independently preregistered.** Its results live under `results/benchmark_d/adaptive_trust_stress/` and are **never pooled** with the primary 75-task pilot-v2 evaluation.

## Question

In the pilot, learned trust moved but never changed the top result, because each role's prior already matched how reliable that role really was. This experiment asks: when an agent's reliability **changes over time on one topic only**, does learned agent/topic trust (`BetaReliability`) change rankings? Does it recover correct information from lower-prior agents, and does it leave the unaffected topic alone?

## What is real and what is controlled

| Layer | Source |
|---|---|
| Environment: files, timeline, the frozen mirror | Generated and deterministic (`experiments/d_env.py`, `generate_stress_tasks`). No LLM. |
| Agent notes | **Real LLM calls**: `gpt-6-luna`, `gpt-6.1-sol`, `grok-4.7`, with the same models, parameters and keys as pilot-v2. |
| Ground truth and verification | Harness checks that read the environment file after the query (`d_env.check`). Never an LLM. |
| Retrieval | Offline replay with retrieval settings pinned to Benchmark C (`retrieval_pinned`) and the same trust parameters as pilot-v2. |

## Environment

- **Two topics in each model's stream:**
  - **`deps`** (degrading): `requirements.lock`.
  - **`config`** (control): `config/app.toml`.
- **The verifier's information source changes, not the verifier.** The verifier is never told to be wrong; the harness only changes which document it sees. From phase 2 on, for `deps` only, it sees a copy from an internal package mirror that stopped syncing at the end of phase 1. The copy is titled `mirror/requirements.lock (internal package mirror)` and its header says `mirror copy, last synced <date>`, so an attentive agent can notice it's out of date.
- **One lower-prior agent per task.** It is the planner or the coder, drawn by the task seed. It writes first (minute 10) and the verifier second (minute 20).
- **Agents are independent.** Neither sees the other's note: the "team notes" section is always `(none)`. Agreement, echoing and repeated claims therefore can't produce the right answer.
- **One fact per task, with a scripted change.** The old value holds until minute 5 and the new value after. The query comes at minute 120; the only trust update is the harness check at minute 130.
- **No pre-query checks.** That means `learned_evidence` is expected to behave about the same as `learned`.

## Task distribution (frozen: `results/benchmark_d/adaptive_trust_stress/tasks.jsonl`, seed `stress-v1`)

All three models get the same 25 tasks in the same order. Phase 1 comes strictly first; the order within each phase is seeded.

| Phase | Scenario | Tasks | Lower agent sees | Verifier sees | Expected right (not forced) |
|---|---|---|---|---|---|
| 1 | `p1_deps_healthy` | 3 | current or stale handbook copy | current | verifier |
| 1 | `p1_config_healthy` | 3 | current or stale handbook copy | current | verifier |
| 2 | `p2_deps_mirror_frozen` | 12 (coder 7, planner 5) | current | frozen mirror | lower agent |
| 2 | `p2_config_control` | 7 (planner 5, coder 2) | stale handbook copy | current | verifier |

**Known limitation:** values come from the shared claim bank, so some "version bumps" are downgrades, such as sqlalchemy 2.0.25 → 1.4.52. In the task they read as a pin change.

## Prompts

The prompts are unchanged from pilot-v2: `prompt_templates_sha256 = 7edc73b6488c87d03ebbb1793f90ba0af184569bfa55398f95df69ca85840155`, the same as in the pilot-v2 manifest. The only difference is that teammate notes are empty, which is the existing template filled with `(none)`.

## Methods

| Method | Description |
|---|---|
| `dense` | Plain vector search (Benchmark C `simple_rag`) |
| `static` | Role-prior trust (`collaborative/full_rerank`) |
| `learned` | `BetaReliability`, per agent and topic |
| `learned_evidence` | `learned` plus evidence-aware retrieval |
| `learned_no_topic` | **Offline diagnostic only**: `BetaReliability(per_topic=False)`, agent-level trust only |

Trust parameters and retrieval weights are exactly those of pilot-v2 and Benchmark C (`d_config.yaml`). Every method replays each model's recorded run with its own fresh database and trust state.

## Metrics

Trust changes and ranking changes are reported separately, because they are different outcomes.

**Incorrect verifier checks before a transition**: the number of post-query harness checks, in earlier tasks of the same model's stream, that judged the verifier's `deps` claim wrong. This comes from the recorded judgements (`memory.reliability.judge` on the real notes), not from the scripted view.

Per model, for `learned` and `learned_no_topic`, each transition below is reported as the first phase-2 `deps` task where it happens, together with that count:

1. **`trust_first_drop_below_prior`**: verifier trust at query time is below its prior (0.95).
2. **`trust_swap_vs_task_lower_agent`**: verifier trust is below the trust of that task's lower agent, on `deps`.
3. **`trust_swap_vs_coder`** and **`trust_swap_vs_planner`**.
4. **`ranking_first_top1_role_differs_from_static`**: the first ranking change.
5. **`first_correct_where_static_wrong`**: the first task where learned is correct@1 and static isn't.

Also reported:

- **Ranking change rate:** the number of tasks where the top result differs from static's, per scenario, for `learned`, `learned_evidence` and `learned_no_topic`.
- **correct@1 and correct@3:** per method, per scenario (phase × topic), pooled and per model, as k/n with Wilson 95% CIs. Paired bootstrap over tasks for `learned − static`, `learned_evidence − learned`, `learned − learned_no_topic` and `learned_evidence − dense`.
- **Lower-prior recovery:** correct@1 on tasks where only the lower-prior agent was right.
- **Topic control:**
  - the verifier's minimum `config` trust across phase 2, and its final value;
  - correct@1 on `p2_config_control` for `learned` vs `static` vs `learned_no_topic`.
- **Agent behaviour:** correct-at-write by scenario × role × view. This shows whether the reliability change actually appeared in the real agents. A verifier that notices the mirror's sync date and answers correctly is a legitimate outcome, not a failure.
- Tokens, cost and latency, as in pilot-v2.

## Expectations (from idealised agents; not predictions of LLM behaviour)

These numbers come from the real `BetaReliability` with the config's parameters, run over the frozen stream, with idealised agents: a current view gives the right value, and a stale or mirror view gives the wrong one. They are asserted by `test_reliability_gives_topic_specific_transitions_on_the_frozen_stream`.

| Transition (incorrect verifier deps checks before it) | `learned` | `learned_no_topic` |
|---|---|---|
| verifier trust below its prior | 1 | 1 |
| verifier below coder on deps | 3 | 5 |
| verifier below planner on deps | 5 | 7 |
| verifier below the task's lower agent | 3 | 7 |
| verifier config trust, minimum in phase 2 | > 0.95 | < 0.5 (falls below the lower agent on config) |

**Topic weighting.** The topic weight is `min(1, (a+b−n0)/3)`. It reaches about 0.9 after phase 1, not 1.0, so `deps` trust still blends in a little agent-level trust.

**Mock run** (offline, mock agents, real pinned embeddings; not evidence):

- **Ranking changes.** The first top-1 change came after 3 incorrect checks with `learned` and after 7 with `learned_no_topic`.
- **`p2_deps_mirror_frozen`, correct@1:**

  | dense | static | learned | learned_no_topic |
  |---|---|---|---|
  | 6/12 | 0/12 | 8/12 | 5/12 |

- **`p2_config_control`, correct@1:** `learned` 7/7, `learned_no_topic` 4/7.
- **So under the frozen retrieval weights, a trust swap can change rankings.**

**What would count against learned trust:**

- `learned` correct@1 on `p2_deps_mirror_frozen` not above `static`;
- no ranking change on `deps` despite a trust swap;
- `learned` doing worse than `static` on `p2_config_control`.

Every one of these will be reported as it falls.

**Uncertainty:** about 12 degraded tasks per model and about 9 after the expected swap, so roughly 27 pooled. Intervals will be wide, roughly ±18 points. Results are exploratory.

## Budget and execution

- **Calls:** 2 per task × 25 tasks × 3 models = 150 calls, plus at most one parse retry each.
- **Hard cap:** $5 for the whole stress run directory. It is checked before every call against recorded spend plus the worst case of the next call.
- **Live calls only after explicit approval.**

Live run:

```
for m in gpt-6-luna grok-4.7 gpt-6.1-sol; do python -m experiments.d_run generate --split stress --model $m --run-id stress-v1 || break; done
```

Offline rescoring:

```
python -m experiments.d_run evaluate --split stress --run-id stress-v1 [--eval-name eval-vN]
```
