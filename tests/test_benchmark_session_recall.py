"""Benchmark scenario smoke test."""

from __future__ import annotations

from benchmarks.scenarios.session_recall import SessionRecallScenario


def test_session_recall_scenario() -> None:
    scenario = SessionRecallScenario()
    scenario.setup()
    metrics = scenario.run()
    scenario.teardown()
    assert metrics.recall_at_k == 1.0
