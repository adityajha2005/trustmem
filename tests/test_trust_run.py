"""Benchmark C: frozen task file and harness behaviour."""

from __future__ import annotations

import json

from experiments import trust_run


def test_task_file_is_frozen_expansion_of_scenarios() -> None:
    expanded = trust_run.expand(trust_run.load_yaml(trust_run.SCENARIOS_PATH), trust_run.load_yaml(trust_run.PARAMS_PATH))
    on_disk = trust_run.load_tasks()
    assert [json.dumps(t, sort_keys=True) for t in expanded] == [json.dumps(t, sort_keys=True) for t in on_disk]


def test_every_task_has_a_true_note() -> None:
    for t in trust_run.load_tasks():
        assert any(n["label"] == "truth" and n["value"] == t["expected"] for n in t["notes"]), t["task_id"]
