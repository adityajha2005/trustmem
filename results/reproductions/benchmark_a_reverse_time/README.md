# Benchmark A: reversed write order (re-scored reproduction)

The README reports that full trustmem stays at 0 of 120 stale when the verifier's note is
written first instead of last. That replay ran inside `experiments/real_run.py noisy`, but its
output was not kept in `results/historical/`. This directory records it.

- Code: tag `baseline-static-trust` (a65c9a7), `experiments.real_run.run_condition(..., reverse_time=True)`.
- Input: the frozen `results/historical/benchmark_a/*/generations.jsonl`. No LLM calls.
- Models: BGE-small-en-v1.5 embeddings, MS MARCO MiniLM cross-encoder, `MEMORY_OFFLINE_RETRIEVAL=0`.
- Generated 2026-11-07, after the historical results were frozen. It is a re-score, not a new run.

Result (stale@1, pooled over 3 models x 40 tasks): dense_rerank 96 of 120, full 0 of 120.
