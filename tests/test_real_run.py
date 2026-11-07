"""Real-run experiment: offline smoke test with the deterministic fake provider."""

from __future__ import annotations

import json

import pytest

from experiments import real_run


def test_parse_notes_tolerates_fences_and_prose() -> None:
    reply = 'Sure!\n```json\n{"key_note": "deploys Wed 09:00", "other_note": "add tests"}\n```'
    assert real_run.parse_notes(reply) == {"key_note": "deploys Wed 09:00", "other_note": "add tests"}
    with pytest.raises(ValueError):
        real_run.parse_notes('{"key_note": "only one"}')


def test_wilson_interval_bounds() -> None:
    lo, hi = real_run.wilson(0, 40)
    assert lo == 0.0 and 0.0 < hi < 0.1
    lo, hi = real_run.wilson(20, 40)
    assert lo < 0.5 < hi


def test_real_run_end_to_end_fake(tmp_path) -> None:
    rc = real_run.main(
        ["all", "--provider", "fake", "--offline", "--limit", "4", "--handcheck-n", "2", "--runs-dir", str(tmp_path)]
    )
    assert rc == 0
    run = tmp_path / "fake"
    assert len((run / "generations.jsonl").read_text().splitlines()) == 4
    summary = json.loads((run / "summary.json").read_text())
    assert set(summary) == set(real_run.CONDITIONS)
    # Trust-aware pipeline must keep the verifier's correction on top.
    assert summary["full"]["stale_at_1"] == 0.0
    assert summary["full"]["correction_at_1"] == 1.0
    assert (run / "handcheck.csv").exists()
    # Generation is resumable: a second run makes no new calls.
    real_run.main(["generate", "--provider", "fake", "--offline", "--limit", "4", "--runs-dir", str(tmp_path)])
    assert len((run / "generations.jsonl").read_text().splitlines()) == 4


def test_noise_flip_swaps_only_coder_and_verifier() -> None:
    flip_all = real_run.noise_settings()["flip_50"]
    tids = [f"t{i}" for i in range(200)]
    flipped = [t for t in tids if flip_all(t, "coder") == 0.95]
    assert 60 < len(flipped) < 140  # roughly half
    for t in flipped:
        assert flip_all(t, "verifier") == 0.70 and flip_all(t, "planner") == 0.60
