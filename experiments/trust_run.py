"""Benchmark C: adaptive trust and evidence-aware retrieval, run as sequential episodes.

    python -m experiments.trust_run tasks      # expand c_scenarios.yaml -> c_tasks.jsonl (design step)
    python -m experiments.trust_run run        # score every condition, write runs/trust_run/

Notes are scripted, so the run is deterministic and needs no API. Within an episode the
order is fixed: agents write, the query is scored with the trust state as it is *now*,
and only then do external sources report the truth for that task. Reliability therefore
only ever learns from earlier tasks.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
PARAMS_PATH = EXP / "c_params.yaml"
SCENARIOS_PATH = EXP / "c_scenarios.yaml"
TASKS_PATH = EXP / "c_tasks.jsonl"
EXPECT_CONTESTED = {"s04_similar_reliability", "s10_conflicting_sources"}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _accuracy(spec: Any, topic: str, index: int) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    if "phases" in spec:
        return float([p for start, p in spec["phases"] if index >= start][-1])
    return float(spec[topic])


def expand(scenarios: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministically expand scenario specs into concrete tasks (one claim per task)."""
    bank = scenarios["claims"]
    tasks: list[dict[str, Any]] = []
    for sc in scenarios["scenarios"]:
        priors = {**params["role_prior"], **sc.get("priors", {})}
        for seed in params["seeds"]:
            rng = random.Random(f"{sc['id']}:{seed}")
            for i in range(sc["tasks"]):
                topic = sc["topics"][i % len(sc["topics"])]
                claim = rng.choice(bank[topic])
                values = rng.sample(claim["values"], 3)
                label_value = {"truth": values[0], "wrong": values[1], "old": values[1], "older": values[2]}
                right = rng.choice(sc["exclusive"]) if "exclusive" in sc else None

                def note(agent: str, key: str, sentence: str, value: str, label: str | None, minutes: int,
                         valid_for: int | None = None) -> dict[str, Any]:
                    return {
                        "agent": agent,
                        "claim_key": key,
                        "value": value,
                        "label": label,  # truth | wrong | old | older | None (distractor)
                        "text": scenarios["note_template"].format(agent=agent, sentence=sentence.format(v=value)),
                        "minutes": minutes,
                        "valid_until_minutes": None if valid_for is None else minutes + valid_for,
                    }

                notes = []
                for pos, w in enumerate(sc["writers"]):
                    if "value" in w:
                        label = w["value"]
                    elif right is not None:
                        label = "truth" if w["agent"] == right else "wrong"
                    else:
                        label = "truth" if rng.random() < _accuracy(w["accuracy"], topic, i) else "wrong"
                    start = w.get("minutes", 10 * pos)
                    for r in range(w.get("repeat", 1)):
                        notes.append(note(w["agent"], claim["key"], claim["sentence"], label_value[label], label,
                                          start + 10 * r, w.get("valid_for")))
                other = rng.choice([c for c in bank[topic] if c["key"] != claim["key"]])
                notes.append(note(rng.choice(sc["writers"])["agent"], other["key"], other["sentence"],
                                  rng.choice(other["values"]), None, 1))

                def events(specs: list[Any]) -> list[dict[str, str]]:
                    out = []
                    for s in specs:
                        kind, value = (s, "truth") if isinstance(s, str) else (s["source"], s["value"])
                        out.append({"source_kind": kind, "source_id": kind, "value": label_value[value]})
                    return out

                used = {n["label"] for n in notes}
                tasks.append(
                    {
                        "scenario": sc["id"],
                        "seed": seed,
                        "index": i,
                        "task_id": f"{sc['id']}-s{seed}-t{i:02d}",
                        "topic": topic,
                        "claim_key": claim["key"],
                        "query": claim["query"],
                        "priors": priors,
                        "expected": values[0],
                        "stale_values": [label_value[x] for x in ("old", "older") if x in used],
                        "expected_contested": sc["id"] in EXPECT_CONTESTED,
                        "notes": notes,
                        "evidence_before": events(sc.get("evidence_before", [])),
                        "verify_after": events(sc.get("verify_after", [])),
                        "query_minutes": scenarios["query_minutes"],
                    }
                )
    return tasks


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


def write_tasks(path: Path = TASKS_PATH) -> Path:
    tasks = expand(load_yaml(SCENARIOS_PATH), load_yaml(PARAMS_PATH))
    with path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tasks"])
    args = parser.parse_args(argv)
    if args.command == "tasks":
        path = write_tasks()
        print(f"{sum(1 for _ in path.open())} tasks -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
