"""Unit tests for evaluation metric primitives."""

from __future__ import annotations

from evaluation.metrics import (
    active_memory_accuracy,
    context_efficiency,
    contradiction_resolution_rate,
    f1_at_k,
    long_horizon_consistency,
    precision_at_k,
    ranking_inversion_rate,
    recall_at_k,
    stale_retrieval_rate,
    suppression_rate,
)


def test_precision_recall_f1_basic() -> None:
    relevant = {"a", "b"}
    retrieved = ["a", "c", "b"]
    assert precision_at_k(relevant, retrieved, k=3) == 2 / 3
    assert recall_at_k(relevant, retrieved, k=3) == 1.0
    assert f1_at_k(relevant, retrieved, k=3) > 0.7


def test_stale_and_active_rates() -> None:
    assert stale_retrieval_rate(["a", "b"], {"a"}) == 0.5
    assert active_memory_accuracy(["a", "b"], {"a", "b"}) == 1.0
    assert active_memory_accuracy([], set()) == 0.0


def test_context_efficiency_uses_full_budget() -> None:
    assert context_efficiency(["x", "y"], {"x", "y"}, k=4) == 0.5
    assert context_efficiency(["x"], {"x"}, k=2) == 0.5


def test_long_horizon_consistency_is_mean() -> None:
    assert long_horizon_consistency([1.0, 0.5, 0.5]) == pytest_approx(2 / 3)


def test_ranking_inversion_rate() -> None:
    # Perfectly calibrated: highest score first
    assert ranking_inversion_rate(["a", "b"], {"a": 0.9, "b": 0.5}) == 0.0
    # Fully inverted
    assert ranking_inversion_rate(["a", "b"], {"a": 0.1, "b": 0.9}) == 1.0


def test_contradiction_resolution_rate() -> None:
    assert contradiction_resolution_rate([True, True, False]) == 2 / 3
    assert contradiction_resolution_rate([]) == 0.0


def test_suppression_rate_bounded() -> None:
    assert suppression_rate(2, 2) == 1.0
    assert suppression_rate(5, 2) == 1.0  # clamps
    assert suppression_rate(0, 0) == 0.0


def pytest_approx(value: float) -> float:
    # Small inline tolerance helper for one assertion above.
    import pytest

    return pytest.approx(value, rel=1e-6)
