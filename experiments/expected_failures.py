"""Expected ablation (variant × scenario) failures — negative controls, not bugs.

These cells are *supposed* to fail scenario pass thresholds. They validate that
benchmarks and ablation toggles behave as designed. Do not treat them as
regressions or attempt to “fix” retrieval to pass them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedFailure:
    variant: str
    scenario: str
    category: str  # e.g. scope_control, trust_ablation, rerank_ablation
    rationale: str
    paper_role: str  # one-line for Results / Ablation section


# Canonical list for 11-variant × 4-scenario grid (results freeze).
EXPECTED_ABLATION_FAILURES: tuple[ExpectedFailure, ...] = (
    ExpectedFailure(
        variant="no_trust_scoring",
        scenario="trust_calibration_v1",
        category="trust_ablation",
        rationale=(
            "Trust weight is zeroed and reallocated to semantic similarity. "
            "The scenario oracle ranks by agent confidence; without a trust "
            "signal, ranking_inversion_rate cannot stay at 0."
        ),
        paper_role=(
            "Confirms the trust-calibration benchmark detects removal of "
            "trust-aware scoring."
        ),
    ),
    ExpectedFailure(
        variant="with_naive_rerank",
        scenario="trust_calibration_v1",
        category="rerank_ablation",
        rationale=(
            "Cross-encoder sorts by lexical query–passage relevance only, "
            "overriding confidence. Planner/coder claims match the query "
            "wording more closely than the verifier’s corrected schedule."
        ),
        paper_role=(
            "Regression baseline for naive reranking: shows why trust-aware "
            "fusion is required when reranking is enabled."
        ),
    ),
    ExpectedFailure(
        variant="no_trust_preservation",
        scenario="trust_calibration_v1",
        category="rerank_ablation",
        rationale=(
            "Fusion combines semantic, trust, and reranker scores but does not "
            "pin high-trust hits (preserve_top_trusted_k=0). Lexical rerank "
            "pressure still inverts the confidence oracle."
        ),
        paper_role=(
            "Isolates trust preservation: fusion weights alone are insufficient "
            "without an explicit calibration guard."
        ),
    ),
    ExpectedFailure(
        variant="local_only",
        scenario="collaborative_recall_v1",
        category="scope_control",
        rationale=(
            "pool_mode=LOCAL_ONLY excludes all SHARED memories. Agents cannot "
            "retrieve cross-agent shared facts (DB, cache, queue), so "
            "shared_recall_at_3 falls below 1.0."
        ),
        paper_role=(
            "Negative control that scope filtering is enforced; collaborative "
            "recall requires hybrid local+shared pools."
        ),
    ),
    ExpectedFailure(
        variant="shared_only",
        scenario="collaborative_recall_v1",
        category="scope_control",
        rationale=(
            "pool_mode=SHARED_ONLY excludes LOCAL memories. Queries targeting "
            "an agent’s private note cannot retrieve the relevant local ID."
        ),
        paper_role=(
            "Negative control for local-scope recall; complements local_only."
        ),
    ),
)


def expected_failure_key(variant: str, scenario: str) -> tuple[str, str]:
    return (variant.strip(), scenario.strip())


def lookup_expected_failure(variant: str, scenario: str) -> ExpectedFailure | None:
    key = expected_failure_key(variant, scenario)
    for entry in EXPECTED_ABLATION_FAILURES:
        if expected_failure_key(entry.variant, entry.scenario) == key:
            return entry
    return None


def is_expected_failure(variant: str, scenario: str) -> bool:
    return lookup_expected_failure(variant, scenario) is not None


def partition_failures(
    runs: list,
) -> tuple[list, list]:
    """Split failed AblationRunMetrics into (expected, unexpected)."""
    expected: list = []
    unexpected: list = []
    for run in runs:
        if run.passed:
            continue
        if is_expected_failure(run.variant, run.scenario):
            expected.append(run)
        else:
            unexpected.append(run)
    return expected, unexpected


def expected_failure_rows() -> list[dict[str, str]]:
    return [
        {
            "variant": e.variant,
            "scenario": e.scenario,
            "category": e.category,
            "paper_role": e.paper_role,
        }
        for e in EXPECTED_ABLATION_FAILURES
    ]


def paper_ablation_discussion_paragraph() -> str:
    """Copy-ready paragraph for an ablation / results write-up."""
    return (
        "Of 44 (variant × scenario) cells per seed, five fail by design. "
        "Scope controls (`local_only`, `shared_only` on collaborative recall) "
        "confirm that pool restrictions are wired correctly. Trust ablations "
        "(`no_trust_scoring`) fail trust calibration when the trust signal is "
        "removed. Rerank ablations (`with_naive_rerank`, `no_trust_preservation`) "
        "fail the same scenario under lexical reranking without trust preservation, "
        "motivating our fusion stage. The production `full` variant passes all "
        "four scenarios including trust calibration when trust-aware fusion and "
        "`preserve_top_trusted_k=1` are enabled. We report raw pass/fail per cell "
        "and do not tune retrieval to satisfy negative controls."
    )
