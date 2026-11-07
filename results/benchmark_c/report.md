# Benchmark C results

Correct top-1 after burn-in (all tasks in brackets).

| Scenario | dense | dense_rerank | static | adaptive | adaptive_no_topic | adaptive_no_decay | ewma | evidence | adaptive_evidence |
|---|---|---|---|---|---|---|---|---|---|
| s01_wrong_verifier | 0.71 (0.74) | 0.54 (0.60) | 0.14 (0.12) | 0.94 (0.80) | 0.94 (0.80) | 0.94 (0.80) | 0.94 (0.86) | 0.14 (0.12) | 0.94 (0.80) |
| s02_reliable_then_unreliable | 0.62 (0.72) | 0.78 (0.83) | 0.33 (0.50) | 0.78 (0.83) | 0.78 (0.83) | 0.78 (0.83) | 0.89 (0.92) | 0.33 (0.50) | 0.78 (0.83) |
| s03_topic_dependent | 0.49 (0.47) | 0.51 (0.47) | 0.44 (0.50) | 0.89 (0.83) | 0.44 (0.50) | 0.89 (0.83) | 1.00 (0.92) | 0.44 (0.50) | 0.89 (0.83) |
| s04_similar_reliability | 0.40 (0.38) | 0.46 (0.46) | 0.43 (0.46) | 0.43 (0.46) | 0.43 (0.46) | 0.43 (0.46) | 0.43 (0.46) | 0.43 (0.46) | 0.43 (0.46) |
| s05_low_trust_verified | 0.60 (0.58) | 0.77 (0.72) | 0.43 (0.46) | 0.89 (0.80) | 0.89 (0.80) | 0.89 (0.80) | 0.97 (0.86) | 0.43 (0.46) | 0.89 (0.80) |
| s06_outdated_with_expiry | 0.57 (0.50) | 0.54 (0.50) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.57 (0.40) | 0.54 (0.52) | 0.54 (0.52) |
| s06b_outdated_no_expiry | 0.49 (0.46) | 0.54 (0.50) | 0.00 (0.00) | 1.00 (0.80) | 1.00 (0.80) | 1.00 (0.80) | 1.00 (0.90) | 0.00 (0.00) | 1.00 (0.80) |
| s07_successive_corrections | 0.03 (0.04) | 0.37 (0.34) | 0.31 (0.32) | 0.31 (0.32) | 0.31 (0.32) | 0.31 (0.32) | 0.31 (0.32) | 1.00 (1.00) | 1.00 (1.00) |
| s08_self_reinforcement | 0.26 (0.28) | 0.77 (0.74) | 0.00 (0.00) | 1.00 (0.90) | 1.00 (0.90) | 1.00 (0.90) | 1.00 (0.90) | 0.00 (0.00) | 1.00 (0.90) |
| s09_no_evidence | 0.66 (0.68) | 0.77 (0.78) | 0.26 (0.24) | 0.26 (0.24) | 0.26 (0.24) | 0.26 (0.24) | 0.26 (0.24) | 0.26 (0.24) | 0.26 (0.24) |
| s10_conflicting_sources | 0.14 (0.16) | 0.60 (0.56) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| s11_evidence_for_lower_trust | 0.31 (0.32) | 0.69 (0.68) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 1.00 (1.00) | 1.00 (1.00) |

## contested_rate

| Scenario | dense | dense_rerank | static | adaptive | adaptive_no_topic | adaptive_no_decay | ewma | evidence | adaptive_evidence |
|---|---|---|---|---|---|---|---|---|---|
| s01_wrong_verifier | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 |
| s02_reliable_then_unreliable | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 |
| s03_topic_dependent | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s04_similar_reliability | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| s05_low_trust_verified | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| s06_outdated_with_expiry | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s06b_outdated_no_expiry | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s07_successive_corrections | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s08_self_reinforcement | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s09_no_evidence | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s10_conflicting_sources | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| s11_evidence_for_lower_trust | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## stale

| Scenario | dense | dense_rerank | static | adaptive | adaptive_no_topic | adaptive_no_decay | ewma | evidence | adaptive_evidence |
|---|---|---|---|---|---|---|---|---|---|
| s01_wrong_verifier | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s02_reliable_then_unreliable | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s03_topic_dependent | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s04_similar_reliability | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s05_low_trust_verified | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s06_outdated_with_expiry | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 | 0.00 | 0.00 |
| s06b_outdated_no_expiry | 0.54 | 0.50 | 1.00 | 0.20 | 0.20 | 0.20 | 0.10 | 1.00 | 0.20 |
| s07_successive_corrections | 0.96 | 0.66 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.00 | 0.00 |
| s08_self_reinforcement | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s09_no_evidence | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s10_conflicting_sources | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| s11_evidence_for_lower_trust | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## wrong_authority

| Scenario | dense | dense_rerank | static | adaptive | adaptive_no_topic | adaptive_no_decay | ewma | evidence | adaptive_evidence |
|---|---|---|---|---|---|---|---|---|---|
| s01_wrong_verifier | 0.02 | 0.32 | 0.88 | 0.16 | 0.16 | 0.16 | 0.10 | 0.88 | 0.16 |
| s02_reliable_then_unreliable | 0.28 | 0.17 | 0.50 | 0.17 | 0.17 | 0.17 | 0.08 | 0.50 | 0.17 |
| s03_topic_dependent | 0.33 | 0.20 | 0.50 | 0.17 | 0.50 | 0.17 | 0.08 | 0.50 | 0.17 |
| s04_similar_reliability | 0.42 | 0.18 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 |
| s05_low_trust_verified | 0.16 | 0.14 | 0.54 | 0.18 | 0.18 | 0.18 | 0.10 | 0.54 | 0.18 |
| s06_outdated_with_expiry | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 | 0.00 | 0.00 |
| s06b_outdated_no_expiry | 0.54 | 0.50 | 1.00 | 0.20 | 0.20 | 0.20 | 0.10 | 1.00 | 0.20 |
| s07_successive_corrections | 0.12 | 0.54 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.00 | 0.00 |
| s08_self_reinforcement | 0.72 | 0.26 | 1.00 | 0.10 | 0.10 | 0.10 | 0.10 | 1.00 | 0.10 |
| s09_no_evidence | 0.02 | 0.18 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 |
| s10_conflicting_sources | 0.84 | 0.44 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| s11_evidence_for_lower_trust | 0.68 | 0.32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |

## inversion_rate

| Scenario | dense | dense_rerank | static | adaptive | adaptive_no_topic | adaptive_no_decay | ewma | evidence | adaptive_evidence |
|---|---|---|---|---|---|---|---|---|---|
| s01_wrong_verifier | 0.42 | 0.53 | 0.88 | 0.29 | 0.29 | 0.29 | 0.20 | 0.88 | 0.29 |
| s02_reliable_then_unreliable | 0.28 | 0.17 | 0.50 | 0.17 | 0.17 | 0.17 | 0.08 | 0.50 | 0.17 |
| s03_topic_dependent | 0.53 | 0.53 | 0.50 | 0.17 | 0.50 | 0.17 | 0.08 | 0.50 | 0.17 |
| s04_similar_reliability | 0.62 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 |
| s05_low_trust_verified | 0.45 | 0.33 | 0.62 | 0.24 | 0.24 | 0.25 | 0.20 | 0.62 | 0.24 |
| s06_outdated_with_expiry | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 | 0.00 | 0.00 |
| s06b_outdated_no_expiry | 0.54 | 0.50 | 1.00 | 0.20 | 0.20 | 0.20 | 0.10 | 1.00 | 0.20 |
| s07_successive_corrections | 0.59 | 0.38 | 0.42 | 0.42 | 0.42 | 0.42 | 0.36 | 0.00 | 0.00 |
| s08_self_reinforcement | 0.00 | 0.00 | 0.08 | 0.01 | 0.01 | 0.01 | 0.01 | 1.00 | 0.10 |
| s09_no_evidence | 0.43 | 0.35 | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 |
| s10_conflicting_sources | 0.84 | 0.44 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| s11_evidence_for_lower_trust | 0.68 | 0.32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |

## brier

| Scenario | dense | dense_rerank | static | adaptive | adaptive_no_topic | adaptive_no_decay | ewma | evidence | adaptive_evidence |
|---|---|---|---|---|---|---|---|---|---|
| s01_wrong_verifier |  |  | 0.383 | 0.209 | 0.209 | 0.200 | 0.173 | 0.383 | 0.209 |
| s02_reliable_then_unreliable |  |  | 0.306 | 0.178 | 0.178 | 0.186 | 0.093 | 0.306 | 0.178 |
| s03_topic_dependent |  |  | 0.275 | 0.169 | 0.293 | 0.160 | 0.089 | 0.275 | 0.169 |
| s04_similar_reliability |  |  | 0.295 | 0.295 | 0.295 | 0.295 | 0.295 | 0.295 | 0.295 |
| s05_low_trust_verified |  |  | 0.316 | 0.230 | 0.230 | 0.227 | 0.218 | 0.316 | 0.230 |
| s06_outdated_with_expiry |  |  | 0.531 | 0.481 | 0.481 | 0.478 | 0.467 | 0.531 | 0.481 |
| s06b_outdated_no_expiry |  |  | 0.531 | 0.197 | 0.197 | 0.178 | 0.104 | 0.531 | 0.197 |
| s07_successive_corrections |  |  | 0.246 | 0.091 | 0.091 | 0.083 | 0.048 | 0.246 | 0.091 |
| s08_self_reinforcement |  |  | 0.325 | 0.121 | 0.121 | 0.109 | 0.064 | 0.325 | 0.121 |
| s09_no_evidence |  |  | 0.352 | 0.352 | 0.352 | 0.352 | 0.352 | 0.352 | 0.352 |
| s10_conflicting_sources |  |  | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 |
| s11_evidence_for_lower_trust |  |  | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 |

## latency_ms

| Scenario | dense | dense_rerank | static | adaptive | adaptive_no_topic | adaptive_no_decay | ewma | evidence | adaptive_evidence |
|---|---|---|---|---|---|---|---|---|---|
| s01_wrong_verifier | 17.8 | 99.4 | 45.2 | 45.0 | 52.1 | 50.5 | 46.7 | 45.4 | 66.6 |
| s02_reliable_then_unreliable | 17.3 | 49.0 | 41.0 | 42.6 | 51.3 | 41.0 | 44.0 | 42.2 | 41.1 |
| s03_topic_dependent | 17.2 | 46.9 | 41.3 | 42.7 | 42.7 | 41.6 | 47.4 | 42.7 | 40.9 |
| s04_similar_reliability | 17.1 | 48.2 | 41.4 | 41.0 | 49.0 | 40.9 | 42.2 | 48.1 | 42.4 |
| s05_low_trust_verified | 18.4 | 52.8 | 45.8 | 45.5 | 45.7 | 52.1 | 49.0 | 45.8 | 46.5 |
| s06_outdated_with_expiry | 21.1 | 47.5 | 41.5 | 44.6 | 42.7 | 41.8 | 42.8 | 40.8 | 39.3 |
| s06b_outdated_no_expiry | 17.5 | 53.1 | 42.5 | 41.7 | 56.0 | 45.2 | 42.0 | 44.1 | 65.5 |
| s07_successive_corrections | 18.5 | 55.7 | 49.6 | 48.9 | 62.5 | 49.3 | 50.5 | 45.5 | 56.2 |
| s08_self_reinforcement | 19.5 | 64.0 | 57.1 | 57.4 | 55.9 | 57.4 | 57.3 | 54.8 | 47.5 |
| s09_no_evidence | 17.8 | 51.0 | 44.8 | 46.3 | 47.0 | 51.2 | 45.2 | 44.9 | 44.7 |
| s10_conflicting_sources | 17.2 | 48.1 | 41.0 | 43.4 | 48.6 | 41.3 | 43.1 | 41.0 | 42.3 |
| s11_evidence_for_lower_trust | 17.1 | 46.7 | 40.9 | 41.1 | 56.9 | 41.2 | 41.1 | 40.6 | 38.5 |

## trust_ms

| Scenario | dense | dense_rerank | static | adaptive | adaptive_no_topic | adaptive_no_decay | ewma | evidence | adaptive_evidence |
|---|---|---|---|---|---|---|---|---|---|
| s01_wrong_verifier | 0.000 | 0.000 | 0.000 | 0.009 | 0.007 | 0.010 | 0.004 | 0.000 | 0.009 |
| s02_reliable_then_unreliable | 0.000 | 0.000 | 0.000 | 0.008 | 0.006 | 0.008 | 0.004 | 0.000 | 0.008 |
| s03_topic_dependent | 0.000 | 0.000 | 0.000 | 0.008 | 0.005 | 0.008 | 0.004 | 0.000 | 0.008 |
| s04_similar_reliability | 0.000 | 0.000 | 0.000 | 0.006 | 0.006 | 0.006 | 0.003 | 0.000 | 0.006 |
| s05_low_trust_verified | 0.000 | 0.000 | 0.000 | 0.009 | 0.006 | 0.009 | 0.004 | 0.000 | 0.009 |
| s06_outdated_with_expiry | 0.000 | 0.000 | 0.000 | 0.008 | 0.005 | 0.007 | 0.003 | 0.000 | 0.007 |
| s06b_outdated_no_expiry | 0.000 | 0.000 | 0.000 | 0.008 | 0.006 | 0.008 | 0.003 | 0.000 | 0.011 |
| s07_successive_corrections | 0.000 | 0.000 | 0.000 | 0.010 | 0.007 | 0.010 | 0.004 | 0.000 | 0.009 |
| s08_self_reinforcement | 0.000 | 0.000 | 0.000 | 0.012 | 0.008 | 0.012 | 0.005 | 0.000 | 0.008 |
| s09_no_evidence | 0.000 | 0.000 | 0.000 | 0.007 | 0.006 | 0.007 | 0.003 | 0.000 | 0.006 |
| s10_conflicting_sources | 0.000 | 0.000 | 0.000 | 0.007 | 0.006 | 0.006 | 0.003 | 0.000 | 0.006 |
| s11_evidence_for_lower_trust | 0.000 | 0.000 | 0.000 | 0.006 | 0.006 | 0.006 | 0.003 | 0.000 | 0.004 |

## Predictions: 57 of 59 hit

| Scenario | Condition | Metric | Predicted | Observed | |
|---|---|---|---|---|---|
| s01_wrong_verifier | static | correct_after_burn_in | [0.0, 0.4] | 0.143 | HIT |
| s01_wrong_verifier | adaptive | correct_after_burn_in | [0.8, 1.0] | 0.943 | HIT |
| s01_wrong_verifier | adaptive_no_topic | correct_after_burn_in | [0.8, 1.0] | 0.943 | HIT |
| s01_wrong_verifier | adaptive_no_decay | correct_after_burn_in | [0.8, 1.0] | 0.943 | HIT |
| s01_wrong_verifier | ewma | correct_after_burn_in | [0.8, 1.0] | 0.943 | HIT |
| s01_wrong_verifier | evidence | same_as static | 0 | 0 | HIT |
| s01_wrong_verifier | adaptive_evidence | correct_after_burn_in | [0.8, 1.0] | 0.943 | HIT |
| s02_reliable_then_unreliable | static | correct_after_burn_in | [0.25, 0.45] | 0.333 | HIT |
| s02_reliable_then_unreliable | adaptive | correct_after_burn_in | [0.65, 1.0] | 0.778 | HIT |
| s02_reliable_then_unreliable | adaptive | correct_from_index_9 | [0.9, 1.0] | 1.0 | HIT |
| s02_reliable_then_unreliable | adaptive_no_decay | correct_from_index_9 | [0.9, 1.0] | 1.0 | HIT |
| s02_reliable_then_unreliable | ewma | correct_from_index_9 | [0.9, 1.0] | 1.0 | HIT |
| s02_reliable_then_unreliable | evidence | same_as static | 0 | 0 | HIT |
| s03_topic_dependent | static | correct_after_burn_in | [0.4, 0.6] | 0.444 | HIT |
| s03_topic_dependent | adaptive | correct_after_burn_in | [0.8, 1.0] | 0.889 | HIT |
| s03_topic_dependent | adaptive_no_topic | correct_after_burn_in | [0.0, 0.6] | 0.444 | HIT |
| s03_topic_dependent | ewma | correct_after_burn_in | [0.8, 1.0] | 1.0 | HIT |
| s04_similar_reliability | static | correct_after_burn_in | [0.2, 0.8] | 0.429 | HIT |
| s04_similar_reliability | static | contested_rate | [0.0, 0.0] | 0.0 | HIT |
| s04_similar_reliability | adaptive | same_as static | 0 | 0 | HIT |
| s04_similar_reliability | evidence | contested_rate | [0.9, 1.0] | 1.0 | HIT |
| s04_similar_reliability | adaptive_evidence | contested_rate | [0.9, 1.0] | 1.0 | HIT |
| s05_low_trust_verified | static | correct_after_burn_in | [0.3, 0.7] | 0.429 | HIT |
| s05_low_trust_verified | adaptive | correct_after_burn_in | [0.8, 1.0] | 0.886 | HIT |
| s05_low_trust_verified | ewma | correct_after_burn_in | [0.8, 1.0] | 0.971 | HIT |
| s06_outdated_with_expiry | static | correct_after_burn_in | [0.0, 0.1] | 0.0 | HIT |
| s06_outdated_with_expiry | adaptive | correct_after_burn_in | [0.0, 0.1] | 0.0 | HIT |
| s06_outdated_with_expiry | evidence | correct_after_burn_in | [0.9, 1.0] | 0.543 | MISS |
| s06_outdated_with_expiry | adaptive_evidence | correct_after_burn_in | [0.9, 1.0] | 0.543 | MISS |
| s06b_outdated_no_expiry | static | correct_after_burn_in | [0.0, 0.1] | 0.0 | HIT |
| s06b_outdated_no_expiry | adaptive | correct_after_burn_in | [0.8, 1.0] | 1.0 | HIT |
| s06b_outdated_no_expiry | evidence | same_as static | 0 | 0 | HIT |
| s07_successive_corrections | static | correct_after_burn_in | [0.1, 0.6] | 0.314 | HIT |
| s07_successive_corrections | adaptive | correct_after_burn_in | [0.1, 0.6] | 0.314 | HIT |
| s07_successive_corrections | evidence | correct_after_burn_in | [0.9, 1.0] | 1.0 | HIT |
| s07_successive_corrections | adaptive_evidence | correct_after_burn_in | [0.9, 1.0] | 1.0 | HIT |
| s08_self_reinforcement | static | correct_after_burn_in | [0.0, 0.1] | 0.0 | HIT |
| s08_self_reinforcement | adaptive | correct_after_burn_in | [0.8, 1.0] | 1.0 | HIT |
| s08_self_reinforcement | ewma | correct_after_burn_in | [0.8, 1.0] | 1.0 | HIT |
| s08_self_reinforcement | evidence | correct_after_burn_in | [0.0, 0.1] | 0.0 | HIT |
| s08_self_reinforcement | adaptive_evidence | correct_after_burn_in | [0.8, 1.0] | 1.0 | HIT |
| s09_no_evidence | static | correct_after_burn_in | [0.0, 0.4] | 0.257 | HIT |
| s09_no_evidence | adaptive | same_as static | 0 | 0 | HIT |
| s09_no_evidence | adaptive_no_topic | same_as static | 0 | 0 | HIT |
| s09_no_evidence | adaptive_no_decay | same_as static | 0 | 0 | HIT |
| s09_no_evidence | ewma | same_as static | 0 | 0 | HIT |
| s09_no_evidence | evidence | same_as static | 0 | 0 | HIT |
| s09_no_evidence | adaptive_evidence | same_as static | 0 | 0 | HIT |
| s10_conflicting_sources | static | contested_rate | [0.0, 0.0] | 0.0 | HIT |
| s10_conflicting_sources | evidence | contested_rate | [0.9, 1.0] | 1.0 | HIT |
| s10_conflicting_sources | adaptive_evidence | contested_rate | [0.9, 1.0] | 1.0 | HIT |
| s11_evidence_for_lower_trust | static | correct_after_burn_in | [0.0, 0.1] | 0.0 | HIT |
| s11_evidence_for_lower_trust | adaptive | same_as static | 0 | 0 | HIT |
| s11_evidence_for_lower_trust | evidence | correct_after_burn_in | [0.9, 1.0] | 1.0 | HIT |
| s11_evidence_for_lower_trust | adaptive_evidence | correct_after_burn_in | [0.9, 1.0] | 1.0 | HIT |
| all (task 0) | adaptive | same_as static | 0 | 0 | HIT |
| all (task 0) | adaptive_no_topic | same_as static | 0 | 0 | HIT |
| all (task 0) | adaptive_no_decay | same_as static | 0 | 0 | HIT |
| all (task 0) | ewma | same_as static | 0 | 0 | HIT |
