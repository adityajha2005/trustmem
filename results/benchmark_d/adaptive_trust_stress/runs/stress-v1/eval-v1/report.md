# Benchmark D results

Rate (k/n) [Wilson 95% CI]. Sequential replay of one recorded agent run per model; 
trust learns only from the post-query harness check. `oracle` = a note with the true value exists.

## correct@1

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | 0.80 (20/25) [0.61, 0.91] | 0.52 (13/25) [0.34, 0.70] | 0.88 (22/25) [0.70, 0.96] | 0.88 (22/25) [0.70, 0.96] | 0.56 (14/25) [0.37, 0.73] |
| gpt-6.1-sol | 0.64 (16/25) [0.45, 0.80] | 0.52 (13/25) [0.34, 0.70] | 0.84 (21/25) [0.65, 0.94] | 0.84 (21/25) [0.65, 0.94] | 0.60 (15/25) [0.41, 0.77] |
| grok-4.7 | 0.76 (19/25) [0.57, 0.89] | 0.52 (13/25) [0.34, 0.70] | 0.84 (21/25) [0.65, 0.94] | 0.84 (21/25) [0.65, 0.94] | 0.60 (15/25) [0.41, 0.77] |
| ALL | 0.73 (55/75) [0.62, 0.82] | 0.52 (39/75) [0.41, 0.63] | 0.85 (64/75) [0.76, 0.92] | 0.85 (64/75) [0.76, 0.92] | 0.59 (44/75) [0.47, 0.69] |

## correct@3

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] |
| gpt-6.1-sol | 0.96 (24/25) [0.80, 0.99] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] |
| grok-4.7 | 0.96 (24/25) [0.80, 0.99] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] |
| ALL | 0.97 (73/75) [0.91, 0.99] | 1.00 (75/75) [0.95, 1.00] | 1.00 (75/75) [0.95, 1.00] | 1.00 (75/75) [0.95, 1.00] | 1.00 (75/75) [0.95, 1.00] |

## stale@1

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | 0.16 (4/25) [0.06, 0.35] | 0.48 (12/25) [0.30, 0.67] | 0.12 (3/25) [0.04, 0.30] | 0.12 (3/25) [0.04, 0.30] | 0.44 (11/25) [0.27, 0.63] |
| gpt-6.1-sol | 0.28 (7/25) [0.14, 0.48] | 0.48 (12/25) [0.30, 0.67] | 0.16 (4/25) [0.06, 0.35] | 0.16 (4/25) [0.06, 0.35] | 0.40 (10/25) [0.23, 0.59] |
| grok-4.7 | 0.20 (5/25) [0.09, 0.39] | 0.48 (12/25) [0.30, 0.67] | 0.16 (4/25) [0.06, 0.35] | 0.16 (4/25) [0.06, 0.35] | 0.40 (10/25) [0.23, 0.59] |
| ALL | 0.21 (16/75) [0.14, 0.32] | 0.48 (36/75) [0.37, 0.59] | 0.15 (11/75) [0.08, 0.24] | 0.15 (11/75) [0.08, 0.24] | 0.41 (31/75) [0.31, 0.53] |

## off_key@1

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | 0.04 (1/25) [0.01, 0.20] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] |
| gpt-6.1-sol | 0.08 (2/25) [0.02, 0.25] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] |
| grok-4.7 | 0.04 (1/25) [0.01, 0.20] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] | 0.00 (0/25) [0.00, 0.13] |
| ALL | 0.05 (4/75) [0.02, 0.13] | 0.00 (0/75) [0.00, 0.05] | 0.00 (0/75) [0.00, 0.05] | 0.00 (0/75) [0.00, 0.05] | 0.00 (0/75) [0.00, 0.05] |

## oracle

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] |
| gpt-6.1-sol | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] |
| grok-4.7 | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] | 1.00 (25/25) [0.87, 1.00] |
| ALL | 1.00 (75/75) [0.95, 1.00] | 1.00 (75/75) [0.95, 1.00] | 1.00 (75/75) [0.95, 1.00] | 1.00 (75/75) [0.95, 1.00] | 1.00 (75/75) [0.95, 1.00] |

## refuted@1 (tasks with pre-query evidence)

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| gpt-6.1-sol | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| grok-4.7 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| ALL | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) |

## correct@1 (lower-prior agent right)

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | 0.67 (8/12) [0.39, 0.86] | 0.00 (0/12) [0.00, 0.24] | 0.75 (9/12) [0.47, 0.91] | 0.75 (9/12) [0.47, 0.91] | 0.42 (5/12) [0.19, 0.68] |
| gpt-6.1-sol | 0.50 (6/12) [0.25, 0.75] | 0.00 (0/12) [0.00, 0.24] | 0.67 (8/12) [0.39, 0.86] | 0.67 (8/12) [0.39, 0.86] | 0.42 (5/12) [0.19, 0.68] |
| grok-4.7 | 0.58 (7/12) [0.32, 0.81] | 0.00 (0/12) [0.00, 0.24] | 0.67 (8/12) [0.39, 0.86] | 0.67 (8/12) [0.39, 0.86] | 0.42 (5/12) [0.19, 0.68] |
| ALL | 0.58 (21/36) [0.42, 0.73] | 0.00 (0/36) [0.00, 0.10] | 0.69 (25/36) [0.53, 0.82] | 0.69 (25/36) [0.53, 0.82] | 0.42 (15/36) [0.27, 0.58] |

## conflict recall

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | 0.00 (0/23) [0.00, 0.14] | 0.00 (0/23) [0.00, 0.14] | 0.00 (0/23) [0.00, 0.14] | 0.04 (1/23) [0.01, 0.21] | 0.00 (0/23) [0.00, 0.14] |
| gpt-6.1-sol | 0.00 (0/23) [0.00, 0.14] | 0.00 (0/23) [0.00, 0.14] | 0.00 (0/23) [0.00, 0.14] | 0.09 (2/23) [0.02, 0.27] | 0.00 (0/23) [0.00, 0.14] |
| grok-4.7 | 0.00 (0/23) [0.00, 0.14] | 0.00 (0/23) [0.00, 0.14] | 0.00 (0/23) [0.00, 0.14] | 0.09 (2/23) [0.02, 0.27] | 0.00 (0/23) [0.00, 0.14] |
| ALL | 0.00 (0/69) [0.00, 0.05] | 0.00 (0/69) [0.00, 0.05] | 0.00 (0/69) [0.00, 0.05] | 0.07 (5/69) [0.03, 0.16] | 0.00 (0/69) [0.00, 0.05] |

## conflict precision

| Model | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| gpt-6-luna | n/a (n=0) | n/a (n=0) | n/a (n=0) | 1.00 (1/1) [0.21, 1.00] | n/a (n=0) |
| gpt-6.1-sol | n/a (n=0) | n/a (n=0) | n/a (n=0) | 1.00 (2/2) [0.34, 1.00] | n/a (n=0) |
| grok-4.7 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 1.00 (2/2) [0.34, 1.00] | n/a (n=0) |
| ALL | n/a (n=0) | n/a (n=0) | n/a (n=0) | 1.00 (5/5) [0.57, 1.00] | n/a (n=0) |

## correct@1 by scenario, all models (descriptive only)

| Scenario | dense | static | learned | learned_evidence | learned_no_topic |
|---|---|---|---|---|---|
| p1_config_healthy | 1.00 (9/9) [0.70, 1.00] | 1.00 (9/9) [0.70, 1.00] | 1.00 (9/9) [0.70, 1.00] | 1.00 (9/9) [0.70, 1.00] | 1.00 (9/9) [0.70, 1.00] |
| p1_deps_healthy | 1.00 (9/9) [0.70, 1.00] | 1.00 (9/9) [0.70, 1.00] | 1.00 (9/9) [0.70, 1.00] | 1.00 (9/9) [0.70, 1.00] | 1.00 (9/9) [0.70, 1.00] |
| p2_config_control | 0.76 (16/21) [0.55, 0.89] | 1.00 (21/21) [0.85, 1.00] | 1.00 (21/21) [0.85, 1.00] | 1.00 (21/21) [0.85, 1.00] | 0.52 (11/21) [0.32, 0.72] |
| p2_deps_mirror_frozen | 0.58 (21/36) [0.42, 0.73] | 0.00 (0/36) [0.00, 0.10] | 0.69 (25/36) [0.53, 0.82] | 0.69 (25/36) [0.53, 0.82] | 0.42 (15/36) [0.27, 0.58] |

## Paired differences in correct@1 (bootstrap over tasks)

| Comparison | n tasks | diff | 95% CI |
|---|---|---|---|
| learned - static (pooled) | 25 | 0.3333 | (0.16, 0.52) |
| learned - static (gpt-6-luna) | 25 | 0.36 | (0.2, 0.56) |
| learned - static (gpt-6.1-sol) | 25 | 0.32 | (0.16, 0.52) |
| learned - static (grok-4.7) | 25 | 0.32 | (0.16, 0.52) |
| learned_evidence - learned (pooled) | 25 | 0.0 | (0.0, 0.0) |
| learned_evidence - learned (gpt-6-luna) | 25 | 0.0 | (0.0, 0.0) |
| learned_evidence - learned (gpt-6.1-sol) | 25 | 0.0 | (0.0, 0.0) |
| learned_evidence - learned (grok-4.7) | 25 | 0.0 | (0.0, 0.0) |
| learned - learned_no_topic (pooled) | 25 | 0.2667 | (0.1067, 0.44) |
| learned - learned_no_topic (gpt-6-luna) | 25 | 0.32 | (0.16, 0.52) |
| learned - learned_no_topic (gpt-6.1-sol) | 25 | 0.24 | (0.08, 0.4) |
| learned - learned_no_topic (grok-4.7) | 25 | 0.24 | (0.08, 0.4) |
| learned_evidence - dense (pooled) | 25 | 0.12 | (-0.04, 0.28) |
| learned_evidence - dense (gpt-6-luna) | 25 | 0.08 | (-0.08, 0.24) |
| learned_evidence - dense (gpt-6.1-sol) | 25 | 0.2 | (-0.08, 0.48) |
| learned_evidence - dense (grok-4.7) | 25 | 0.08 | (-0.12, 0.28) |

## Agents (real LLM calls)

| Model | calls | prompt tok | completion tok | reasoning tok | cost USD | p50 ms | parse failed | wrong key |
|---|---|---|---|---|---|---|---|---|
| gpt-6-luna | 55 | 18939 | 8226 | 4214 | 0.0060 | 2097.7 | 0.00 (0/54) [0.00, 0.07] | 0.00 (0/54) [0.00, 0.07] |
| gpt-6.1-sol | 50 | 17221 | 3852 | 67 | 0.0730 | 1943.4 | 0.00 (0/50) [0.00, 0.07] | 0.00 (0/50) [0.00, 0.07] |
| grok-4.7 | 50 | 79717 | 5026 | 9677 | 0.1896 | 3381.4 | 0.00 (0/50) [0.00, 0.07] | 0.00 (0/50) [0.00, 0.07] |

# Adaptive-trust stress analysis

Counts in brackets: incorrect verifier checks on deps (post-query harness checks) before the transition.

## gpt-6-luna

| Transition | learned | learned_no_topic |
|---|---|---|
| trust_first_drop_below_prior | stress-007-p2_deps_mirror_frozen [2] | stress-007-p2_deps_mirror_frozen [2] |
| trust_swap_vs_task_lower_agent | stress-010-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [8] |
| trust_swap_vs_coder | stress-007-p2_deps_mirror_frozen [2] | stress-013-p2_deps_mirror_frozen [5] |
| trust_swap_vs_planner | stress-014-p2_deps_mirror_frozen [6] | stress-016-p2_deps_mirror_frozen [8] |
| ranking_first_top1_role_differs_from_static | stress-010-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [8] |
| first_correct_where_static_wrong | stress-010-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [8] |

Top-1 differs from static (k/n tasks):

| Scenario | learned | learned_evidence | learned_no_topic |
|---|---|---|---|
| p1_config_healthy | 0/3 | 0/3 | 0/3 |
| p1_deps_healthy | 0/3 | 0/3 | 0/3 |
| p2_config_control | 0/7 | 0/7 | 4/7 |
| p2_deps_mirror_frozen | 9/12 | 9/12 | 5/12 |

Topic control: `{"learned": {"min_verifier_config_trust_phase2": 0.946318, "verifier_config_trust_end": 0.980976, "verifier_deps_trust_end": 0.379017}, "learned_no_topic": {"min_verifier_config_trust_phase2": 0.438502, "verifier_config_trust_end": 0.634843, "verifier_deps_trust_end": 0.634843}}`

Verifier judged wrong on deps: 13 checks

## gpt-6.1-sol

| Transition | learned | learned_no_topic |
|---|---|---|
| trust_first_drop_below_prior | stress-007-p2_deps_mirror_frozen [1] | stress-007-p2_deps_mirror_frozen [1] |
| trust_swap_vs_task_lower_agent | stress-012-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [7] |
| trust_swap_vs_coder | stress-012-p2_deps_mirror_frozen [3] | stress-014-p2_deps_mirror_frozen [5] |
| trust_swap_vs_planner | stress-014-p2_deps_mirror_frozen [5] | stress-016-p2_deps_mirror_frozen [7] |
| ranking_first_top1_role_differs_from_static | stress-012-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [7] |
| first_correct_where_static_wrong | stress-012-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [7] |

Top-1 differs from static (k/n tasks):

| Scenario | learned | learned_evidence | learned_no_topic |
|---|---|---|---|
| p1_config_healthy | 0/3 | 0/3 | 0/3 |
| p1_deps_healthy | 0/3 | 0/3 | 0/3 |
| p2_config_control | 0/7 | 0/7 | 3/7 |
| p2_deps_mirror_frozen | 8/12 | 8/12 | 5/12 |

Topic control: `{"learned": {"min_verifier_config_trust_phase2": 0.952137, "verifier_config_trust_end": 0.980976, "verifier_deps_trust_end": 0.382326}, "learned_no_topic": {"min_verifier_config_trust_phase2": 0.44541, "verifier_config_trust_end": 0.641264, "verifier_deps_trust_end": 0.641264}}`

Verifier judged wrong on deps: 12 checks

## grok-4.7

| Transition | learned | learned_no_topic |
|---|---|---|
| trust_first_drop_below_prior | stress-007-p2_deps_mirror_frozen [1] | stress-007-p2_deps_mirror_frozen [1] |
| trust_swap_vs_task_lower_agent | stress-012-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [7] |
| trust_swap_vs_coder | stress-012-p2_deps_mirror_frozen [3] | stress-014-p2_deps_mirror_frozen [5] |
| trust_swap_vs_planner | stress-014-p2_deps_mirror_frozen [5] | stress-016-p2_deps_mirror_frozen [7] |
| ranking_first_top1_role_differs_from_static | stress-012-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [7] |
| first_correct_where_static_wrong | stress-012-p2_deps_mirror_frozen [3] | stress-016-p2_deps_mirror_frozen [7] |

Top-1 differs from static (k/n tasks):

| Scenario | learned | learned_evidence | learned_no_topic |
|---|---|---|---|
| p1_config_healthy | 0/3 | 0/3 | 0/3 |
| p1_deps_healthy | 0/3 | 0/3 | 0/3 |
| p2_config_control | 0/7 | 0/7 | 3/7 |
| p2_deps_mirror_frozen | 8/12 | 8/12 | 5/12 |

Topic control: `{"learned": {"min_verifier_config_trust_phase2": 0.952137, "verifier_config_trust_end": 0.980976, "verifier_deps_trust_end": 0.382326}, "learned_no_topic": {"min_verifier_config_trust_phase2": 0.44541, "verifier_config_trust_end": 0.641264, "verifier_deps_trust_end": 0.641264}}`

Verifier judged wrong on deps: 12 checks

