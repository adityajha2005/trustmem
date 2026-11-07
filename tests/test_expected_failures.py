"""Tests for expected ablation failure registry."""

from __future__ import annotations

from evaluation.ablation_metrics import AblationRunMetrics
from experiments.expected_failures import (
    EXPECTED_ABLATION_FAILURES,
    is_expected_failure,
    partition_failures,
)


def test_registry_has_five_entries() -> None:
    assert len(EXPECTED_ABLATION_FAILURES) == 5


def test_full_trust_is_not_expected_failure() -> None:
    assert not is_expected_failure("full", "trust_calibration_v1")


def test_naive_rerank_trust_is_expected() -> None:
    assert is_expected_failure("with_naive_rerank", "trust_calibration_v1")


def test_partition_failures() -> None:
    runs = [
        AblationRunMetrics("full", "trust_calibration_v1", passed=False),
        AblationRunMetrics("with_naive_rerank", "trust_calibration_v1", passed=False),
        AblationRunMetrics("full", "long_horizon_v1", passed=False),
    ]
    expected, unexpected = partition_failures(runs)
    assert len(expected) == 1
    assert len(unexpected) == 2
    assert unexpected[0].variant == "full"
