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


def test_offline_episode_invariants() -> None:
    """Hash embeddings: checks that hold whatever the embedding model is."""
    params = trust_run.load_yaml(trust_run.PARAMS_PATH)
    tasks = [t for t in trust_run.load_tasks() if t["seed"] == 0 and t["scenario"] in
             ("s01_wrong_verifier", "s09_no_evidence", "s11_evidence_for_lower_trust")]
    rows = []
    for cond in ("static", "adaptive", "evidence"):
        rows += trust_run.run_condition(cond, tasks, params)
    out = {(r["condition"], r["task_id"]): r for r in rows}
    for t in tasks:
        static = out[("static", t["task_id"])]["outcome"]
        if t["scenario"] == "s09_no_evidence" or t["index"] == 0:
            # no verification history -> adaptive trust is exactly static trust
            assert out[("adaptive", t["task_id"])]["outcome"] == static
        if t["scenario"] == "s11_evidence_for_lower_trust":
            assert static == "wrong" and out[("evidence", t["task_id"])]["outcome"] == "correct"
    s01 = [r for r in rows if r["scenario"] == "s01_wrong_verifier" and r["index"] >= params["burn_in"]]
    late = lambda c: sum(r["outcome"] == "correct" for r in s01 if r["condition"] == c)  # noqa: E731
    assert late("adaptive") > late("static")
    checks = trust_run.check_predictions(rows, trust_run.load_yaml(trust_run.PREDICTIONS_PATH), params["burn_in"])
    assert checks and all({"hit", "observed", "predicted"} <= c.keys() for c in checks)
