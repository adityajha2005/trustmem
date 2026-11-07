"""Benchmark B: task-file validation and offline scoring with the fake provider."""

from __future__ import annotations

import json

from experiments import hard_run


def test_task_file_is_valid_and_balanced() -> None:
    tasks = hard_run.load_tasks()
    assert hard_run.validate(tasks) == []
    counts = {c: sum(t["category"] == c for t in tasks) for c in hard_run.CATEGORIES}
    assert all(n >= 7 for n in counts.values()), counts


def test_validate_catches_bad_labels() -> None:
    bad = {
        "id": "x", "category": "ambiguous", "context": "c", "why": "w", "query": "q",
        "facts": {"a": "A", "b": "B"},
        "writes": [{"role": "planner", "fact": "a", "kind": "build"}],
        "expected": "c",
    }
    errors = hard_run.validate([bad])
    assert any("expected" in e for e in errors)
    assert any("never written" in e for e in errors)
    assert any("previous write" in e for e in errors)


def test_prompt_refers_to_previous_write() -> None:
    task = next(t for t in hard_run.load_tasks() if t["id"] == "wv_rate_limit")
    notes = [{"key_note": "limit is 1000/min"}, {"key_note": "building for 1000/min"}]
    prompt = hard_run.prompt_for(task, 2, notes)
    # The verifier's `correct` prompt names the earlier fact and its own (wrong) one.
    assert task["facts"]["right"] in prompt and task["facts"]["wrong"] in prompt
    assert '"limit is 1000/min"' in hard_run.prompt_for(task, 1, notes)  # coder builds on planner


def test_fake_run_scores_against_expected(tmp_path) -> None:
    rc = hard_run.main(["all", "--provider", "fake", "--offline", "--runs-dir", str(tmp_path)])
    assert rc == 0
    run = tmp_path / "fake"
    results = [json.loads(line) for line in (run / "results.jsonl").open()]
    tasks = {t["id"]: t for t in hard_run.load_tasks()}
    for r in results:
        # The outcome must agree with the label of the top hit and the predefined answer.
        expected = tasks[r["task_id"]]["expected"]
        if r["top_fact"] is None:
            assert r["outcome"] == "miss"
        else:
            assert r["outcome"] == ("correct" if r["top_fact"] == expected else "wrong")
    # Trust pins the verifier, so a wrong verifier wins under the full pipeline.
    wv = [r for r in results if r["category"] == "wrong_verifier" and r["condition"] == "full"]
    assert wv and all(r["top_role"] == "verifier" for r in wv)
    assert (run / "label_check.csv").exists()
