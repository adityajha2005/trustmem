# Benchmark C: adaptive trust and evidence-aware retrieval

Benchmark C is new work, separate from Benchmarks A and B, which stay frozen in `results/historical/`. This file holds the design. It was committed together with the task file and the predictions, before any method code existed. Results are added in a later, separate commit.

## Question

Benchmark B showed that static role trust follows the most trusted agent even when it is wrong (0 of 16 wrong-verifier runs). Benchmark C asks two things. Can trust learned from independently verified outcomes fix that without breaking the cases static trust gets right? And does evidence attached to claims help where trust can't?

## Files

| File | Role |
|---|---|
| `experiments/c_params.yaml` | Frozen parameters: role priors, n₀ = 4, λ = 0.90, m = 3, δ = 0.05, EWMA rate 0.30, burn-in 3, seeds 0 to 4 |
| `experiments/c_scenarios.yaml` | 12 scenarios and the claim bank (3 topics × 4 claims × 3 values) |
| `experiments/c_tasks.jsonl` | 620 expanded tasks with their expected answers. The harness reads only this file |
| `experiments/c_predictions.yaml` | Predicted outcome ranges per scenario and condition |
| `experiments/trust_run.py` | `tasks` regenerates the task file deterministically; `run` scores it |

## Episode protocol

One scenario and one seed make an episode of 10 or 12 tasks. Every task is one claim (a `claim_key`, such as `api.rate_limit`) in its own task scope, plus one distractor note about a different claim on the same topic. For each task, in order:

1. Agents write their notes. The text is scripted: `Note from the {agent}: {sentence}.`
2. Any `evidence_before` events are recorded (used in scenarios 10 and 11 only).
3. The query is scored with the trust state as it is at that moment.
4. The `verify_after` sources report the true value, and only then are the agents' reliability estimates updated.

The expected answer for a task is never available before that task is scored, so reliability learns only from earlier tasks. Notes are not shared across tasks, only the reliability state is.

## Conditions

| Condition | Trust | Evidence and claim groups |
|---|---|---|
| `dense` | none (plain vector search) | no |
| `dense_rerank` | none (+ MS MARCO cross-encoder) | no |
| `static` | role prior, as in Benchmarks A and B | no |
| `adaptive` | Beta reliability per agent and topic | no |
| `adaptive_no_topic` | Beta reliability per agent only | no |
| `adaptive_no_decay` | Beta reliability with λ = 1 | no |
| `ewma` | EWMA per agent and topic, starting at the role prior | no |
| `evidence` | role prior | yes |
| `adaptive_evidence` | Beta reliability per agent and topic | yes |

All conditions apart from `dense` and `dense_rerank` use the default trustmem pipeline: the trust-aware fusion with the most trusted hit pinned.

## Reliability rules (frozen)

- **Prior.** For agent a with role trust r_a: α₀ = n₀·r_a and β₀ = n₀·(1 − r_a), with n₀ = 4. The same prior is used at agent level and for each (agent, topic).
- **Update.** An update happens only when a verification event comes from an external source (`test`, `config` or `human`). Source kind `agent` is stored but never changes reliability. The update discounts the accumulated evidence toward the prior: α ← λα + (1 − λ)α₀, and the same for β. It then adds 1 to α if the claim was right, or 1 to β if it was wrong. λ applies only to accumulated verification evidence. Writes and retrievals never decay anything.
- **What gets judged.** One outcome per agent per event. It is that agent's latest claim on the event's `claim_key` that hadn't expired when the event happened. Repeating a claim adds nothing, an agent that corrected itself is judged on its final answer, and an expired claim is not judged at all.
- **Trust used for ranking.** Write e_t for the evidence mass on the (agent, topic) pair, which is α + β − n₀. Then w = min(1, e_t / m) and trust = w · mean(agent, topic) + (1 − w) · mean(agent). The topic is the `claim_key` prefix before the first dot.
- **Never updated from** retrieval results, agreement between agents, or an agent's own confidence.
- **EWMA baseline.** For each (agent, topic): t ← (1 − 0.3)·t + 0.3·outcome, starting at the role prior, with no backoff.

## Evidence rules (frozen)

- Only external events count. For each `claim_key`, the latest round (the events with the newest timestamp) gives the verdict.
  - If every event in that round reports the same value, that value is verified.
  - If they disagree, the claim is contested and nothing is supported or refuted.
- **Stage 4 (lineage filter)** drops three kinds of note:
  - a note whose claim was refuted by a verdict made after the note was written;
  - a note past its `valid_until` at query time;
  - an agent's older notes on a `claim_key` it has written again, so only its latest note remains.
- **Trust.** A note whose value is verified gets trust 1.0. Other notes keep their trust from the condition (static or adaptive).
- **Contested flag.** It is set when external sources disagree. It is also set when two or more different values remain for the queried claim and the best trust of the top two values differs by less than δ = 0.05. It is reported in the retrieval diagnostics. It doesn't change the order.

## Metrics

- **Outcomes:** correct@1 over all tasks and after the 3-task burn-in.
  - **stale@1:** the top note states a value that was true earlier (scenarios 6, 6b and 7).
  - **wrong-authority@1:** the top note is wrong and comes from the agent with the highest prior in that task.
- **Ranking inversions:** the share of (correct, wrong) note pairs in the top 5 where the wrong note ranks higher.
- **Contested rate.**
- **Trust calibration:** the Brier score of each writer's trust at query time against whether that writer's claim in the task was right. Only for conditions that use trust.
- **Cost:** mean retrieval latency, and the time spent in the trust and evidence code.
- **Statistics:** about 30 to 60 scored tasks per scenario and condition, all written by the author. The results are exploratory patterns, not significance tests.

## Hand calculations behind the predictions

Using n₀ = 4 and λ = 0.9:

- **Scenario 1 (wrong verifier).** The verifier starts at (3.8, 0.2), a mean of 0.95.
  - After one wrong outcome it is at (3.8, 1.2), 0.76.
  - After two it is at (3.8, 2.1), 0.64.
  - The planner, after two right outcomes, is at (4.3, 1.6), 0.73.
  - So the planner overtakes after about two verifier errors, inside the burn-in.
- **Scenario 2 (verifier turns wrong at task 6).** After six right outcomes the verifier is at about 0.98 and the planner at about 0.82.
  - After one wrong the verifier is at 0.87 against the planner's 0.83, so it still wins.
  - After two wrongs it is at 0.78 against 0.84, and the planner wins.
  - So tasks 6 and 7 are wrong, and everything from task 8 on is right.
- **Scenario 3 (topics).** With per-topic trust, the coder on infra after one wrong outcome blends to 0.605, against the planner's 0.611. Right from the second infra task.
  - Agent-level trust only sees two agents who are each right half the time, and their last outcomes alternate with the topic. That's why it's predicted at or below static.
- **Scenario 6 (with expiry).** The verifier's expired note is never judged, so the verifier stays at 0.95. The planner tops out near 12.4 / 14 ≈ 0.89. Adaptive trust alone stays stale, and only expiry fixes it.
- **Scenario 6b (no expiry).** The verifier's latest claim is the old value, so it is judged wrong after the world changed. Adaptive trust "wins", but by penalising an agent that wasn't wrong when it wrote the note. This is a predicted failure mode, not a success.

## Process record

| Commit | What |
|---|---|
| `baseline-static-trust` (tag, `a65c9a7`) | Code that produced Benchmarks A and B |
| `e5245bd` | A and B recorded runs frozen in `results/historical/`, with a re-scoring regression test |
| `a490ca8` | This design: parameters, scenarios, 620 tasks, predictions. No method code existed yet |
| `aaa83a4` | Reliability, evidence, trust seam, harness, unit tests |
| `d822e58` | Results of the real-model run (`results/benchmark_c/`) |

Before the method commit, I ran the harness once offline with hash embeddings as a smoke test. That run changed one thing: the prediction checker crashed on a scenario subset, so it now skips scenarios that have no rows. The method, the parameters, the tasks and the predictions were not changed after any result was seen. The offline and real runs give the same top-1 outcomes for every trust condition, because the pinned top note is chosen by trust, not by the embeddings.

## Reproducing

```bash
python -m experiments.trust_run tasks   # regenerates c_tasks.jsonl; must be byte-identical (tested)
python -m experiments.trust_run run     # about 5 minutes on CPU with BGE + MiniLM cached
pytest tests/test_trust_run.py tests/test_reliability_evidence.py   # offline, hash embeddings
TRUSTMEM_REAL_MODELS=1 pytest tests/test_historical_results.py      # A and B still re-score identically
```

`results/benchmark_c/manifest.json` records the commit, the Python version, the models, and SHA-256 hashes of the four input files. There are no LLM calls and no API cost.

## Results (real run, `d822e58`)

Correct top-1 after the 3-task burn-in, pooled over 5 seeds (35 to 45 tasks per cell). Tasks within an episode share trust state, so they are not independent. These are exploratory patterns, not significance tests.

| Scenario | dense | +rerank | static | adaptive | no topic | EWMA | evidence | adaptive + evidence |
|---|---|---|---|---|---|---|---|---|
| 1 wrong verifier | 0.71 | 0.54 | 0.14 | 0.94 | 0.94 | 0.94 | 0.14 | 0.94 |
| 2 reliable, then unreliable | 0.62 | 0.78 | 0.33 | 0.78 | 0.78 | 0.89 | 0.33 | 0.78 |
| 3 reliability varies by topic | 0.49 | 0.51 | 0.44 | 0.89 | 0.44 | 1.00 | 0.44 | 0.89 |
| 4 similar reliability (contested flag) | 0.40 | 0.46 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 (flag 1.00) | 0.43 (flag 1.00) |
| 5 low-trust agent verified right | 0.60 | 0.77 | 0.43 | 0.89 | 0.89 | 0.97 | 0.43 | 0.89 |
| 6 outdated, with expiry | 0.57 | 0.54 | 0.00 | 0.00 | 0.00 | 0.57 | 0.54 | 0.54 |
| 6b outdated, no expiry | 0.49 | 0.54 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| 7 successive corrections | 0.03 | 0.37 | 0.31 | 0.31 | 0.31 | 0.31 | 1.00 | 1.00 |
| 8 self-reinforcement | 0.26 | 0.77 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| 9 no evidence (control) | 0.66 | 0.77 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| 10 conflicting sources (contested flag) | 0.14 | 0.60 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 (flag 1.00) | 0.00 (flag 1.00) |
| 11 evidence for the lower-trust agent | 0.31 | 0.69 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

`adaptive_no_decay` matched `adaptive` in every scenario, so it's left out of this table. The full tables also cover all-task accuracy, stale and wrong-authority rates, inversions, Brier scores, latency and the per-prediction checks, and are in `results/benchmark_c/report.md`.

What the run shows, including where it fails:

- **Learned trust fixes static trust's failure, once verified history exists.** In scenario 1, static trust was right on 0.14 of tasks and learned trust on 0.94. Scenarios 2, 3, 5 and 8 show the same pattern.
  - The gain depends on three things: verification outcomes from earlier tasks, external sources that are always right in this benchmark, and writers that label `claim_key` and `claim_value` correctly.
  - Benchmark B has none of these. With no history, learned trust is identical to static trust by construction (the cold-start and scenario 9 checks pass exactly). So B's 0 of 16 is not fixed in B's setting.
- **Without evidence, trusting the wrong agent is worse than plain search.**
  - In scenario 9, dense got 0.66 and dense + rerank 0.77, against 0.26 for every trust condition.
  - In scenario 10, dense + rerank got 0.60 against 0.
  - Dense retrieval roughly follows the share of notes that are right. A trust prior concentrates on one agent, which is right or wrong as a whole.
- **EWMA matched or beat the Beta model.** It did so on scenarios 2, 3 and 5, and had a lower Brier score in most scenarios.
  - Turning off decay changed nothing, because episodes of 10 to 12 tasks are too short for it to matter.
  - Nothing in this benchmark tests the reason for choosing Beta (a single mistake can only move trust so far). These results don't show that Beta is the better choice.
- **Evidence and claim groups help where trust can't.**
  - Scenario 7, successive corrections: 1.00 against 0.31 for static.
  - Scenario 11, evidence for the lower-trust agent: 1.00 against 0.
  - Similar reliability (scenario 4) and conflicting sources (scenario 10) are flagged as contested in every task.
  - Each of these relies on correct `claim_key` and `claim_value` labels, which the LLM agents in A and B never produced.
- **Two predictions missed. Both are in scenario 6, for the evidence and adaptive + evidence conditions: 0.54 against a predicted 0.9 to 1.0.**
  - The expired note is dropped correctly.
  - In all 24 misses, the pin guard then lifts an irrelevant distractor note written by the verifier above the right answer. That's the known pinning flaw, triggered here by how distractors are assigned.
  - It is reported as a miss and was not fixed and rerun.
- **Scenario 6b succeeds for the wrong reason, as predicted.** The verifier is penalised for a fact that changed after it wrote its note. Learned trust can't tell "was wrong" from "became outdated" without validity information.
- **Adaptive + evidence flags 8 to 10% of tasks as contested in scenarios 1, 2 and 5.** That happens when two agents' learned trust passes within 0.05 of each other. These flags weren't predicted.
- **Cost.** The trust lookup adds about 0.01 ms per query. The latency differences between the trustmem conditions (about 40 to 65 ms, mostly the cross-encoder) are noise.
