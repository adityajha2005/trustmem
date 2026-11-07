"""Benchmarks A and B are historical baselines: frozen artifacts, checked two ways.

1. Offline (always): the recorded files are byte-identical to the frozen checksums, and
   they still contain the numbers the README and proposal quote.
2. With real models (TRUSTMEM_REAL_MODELS=1): re-scoring the recorded LLM generations with
   the current code reproduces the recorded results exactly. This never calls an LLM API;
   fresh generations would differ, so they are not the reference.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from experiments import hard_run, real_run

HIST = Path(__file__).resolve().parents[1] / "results" / "historical"
A_MODELS = ["grok-4.7", "gpt-6-luna", "gpt-6.1-sol"]
B_MODELS = ["grok-4.7", "gpt-6.1-sol"]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _canon(rows: list[dict]) -> list[str]:
    return sorted(json.dumps(r, sort_keys=True) for r in rows)


def test_artifacts_match_checksums() -> None:
    for line in (HIST / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((HIST / name).read_bytes()).hexdigest() == digest, name


def test_benchmark_a_published_numbers() -> None:
    stale = {c: [] for c in ("dense", "dense_rerank", "full")}
    for m in A_MODELS:
        rows = _rows(HIST / "benchmark_a" / m / "results.jsonl")
        for c in stale:
            stale[c].append(sum(r["stale_at_1"] for r in rows if r["condition"] == c))
    assert stale["dense"] == [27, 35, 31]  # 93/120 = 77.5%
    assert stale["dense_rerank"] == [28, 33, 35]  # 96/120 = 80%
    assert stale["full"] == [0, 0, 0]


def test_benchmark_b_wrong_verifier_numbers() -> None:
    correct = {c: 0 for c in ("dense", "full")}
    for m in B_MODELS:
        for r in _rows(HIST / "benchmark_b" / m / "results.jsonl"):
            if r["category"] == "wrong_verifier" and r["condition"] in correct:
                correct[r["condition"]] += r["outcome"] == "correct"
    assert correct == {"dense": 9, "full": 0}  # of 16


@pytest.mark.skipif(os.getenv("TRUSTMEM_REAL_MODELS") != "1", reason="needs cached BGE + MiniLM; set TRUSTMEM_REAL_MODELS=1")
@pytest.mark.parametrize(
    ("bench", "model"), [("a", m) for m in A_MODELS] + [("b", m) for m in B_MODELS]
)
def test_rescoring_reproduces_recorded_results(bench: str, model: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_OFFLINE_RETRIEVAL", "0")
    monkeypatch.delenv("PYTEST_CURRENT_TEST")  # make_bench_config forces hash embeddings under pytest
    src = HIST / f"benchmark_{bench}" / model
    shutil.copy(src / "generations.jsonl", tmp_path / "generations.jsonl")
    mod = real_run if bench == "a" else hard_run
    mod.evaluate(tmp_path, mod.load_tasks())
    assert _canon(_rows(tmp_path / "results.jsonl")) == _canon(_rows(src / "results.jsonl"))
