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
import hashlib
import json
import platform
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
PARAMS_PATH = EXP / "c_params.yaml"
SCENARIOS_PATH = EXP / "c_scenarios.yaml"
TASKS_PATH = EXP / "c_tasks.jsonl"
PREDICTIONS_PATH = EXP / "c_predictions.yaml"
RESULTS_DIR = ROOT / "results" / "benchmark_c"
EPOCH = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)  # task i of an episode happens on day i
EVIDENCE_BEFORE_MIN = 80  # after every note (latest at minute 60), before the query (90)
VERIFY_AFTER_MIN = 100  # after the query

# condition -> (technique, ablation, reliability kind, evidence on)
CONDITIONS: dict[str, tuple[str, str | None, str | None, bool]] = {
    "dense": ("simple_rag", None, None, False),
    "dense_rerank": ("collaborative", "with_naive_rerank", None, False),
    "static": ("collaborative", "full_rerank", None, False),
    "adaptive": ("collaborative", "full_rerank", "beta", False),
    "adaptive_no_topic": ("collaborative", "full_rerank", "beta_no_topic", False),
    "adaptive_no_decay": ("collaborative", "full_rerank", "beta_no_decay", False),
    "ewma": ("collaborative", "full_rerank", "ewma", False),
    "evidence": ("collaborative", "full_rerank", None, True),
    "adaptive_evidence": ("collaborative", "full_rerank", "beta", True),
}
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def make_reliability(kind: str | None, priors: dict[str, float], params: dict[str, Any]):
    from memory.reliability import BetaReliability, EWMAReliability

    if kind is None:
        return None
    if kind == "ewma":
        return EWMAReliability(priors, rate=params["ewma_rate"])
    return BetaReliability(
        priors,
        prior_strength=params["prior_strength"],
        retention=1.0 if kind == "beta_no_decay" else params["retention"],
        topic_threshold=params["topic_threshold"],
        per_topic=kind != "beta_no_topic",
    )


def _when(task: dict[str, Any], minutes: float) -> datetime:
    return EPOCH + timedelta(days=task["index"], minutes=minutes)


def _events(task: dict[str, Any], key: str, minutes: int):
    from memory.evidence import VerificationEvent

    return [VerificationEvent(claim_key=task["claim_key"], at=_when(task, minutes), **e) for e in task[key]]


def _episodes(tasks: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    eps: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        eps[(t["scenario"], t["seed"])].append(t)
    return {k: sorted(v, key=lambda t: t["index"]) for k, v in eps.items()}


def run_condition(condition: str, tasks: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Insert every task's notes, then walk each episode in order: query, then learn."""
    from baselines.factory import create_backend
    from baselines.runtime import configure_torch_safety
    from benchmarks._helpers import make_bench_config, reset_storage_engine
    from experiments.real_run import _ablation
    from memory.evidence import Evidence, topic_of
    from memory.reliability import learn
    from memory.schema import Memory, MemoryScope
    from retrieval.context import RetrievalContext
    from storage import init_db, session_scope
    from storage.repository import MemoryRepository
    from storage.sqlite_store import SQLiteMemoryStore

    technique, ablation, rel_kind, use_evidence = CONDITIONS[condition]
    configure_torch_safety()
    reset_storage_engine()
    tmp = Path(tempfile.mkdtemp(prefix=f"trust_run_{condition}_"))
    rows: list[dict[str, Any]] = []
    try:
        config = make_bench_config(f"trust_run_{condition}", faiss_dir=tmp)
        init_db(config)
        with session_scope(config) as session:
            store = SQLiteMemoryStore(session)
            backend = create_backend(technique, MemoryRepository(session), config, work_dir=tmp, ablation=_ablation(ablation))
            labels: dict[str, dict[str, Any]] = {}
            memories: dict[str, list[Memory]] = {}
            for t in tasks:
                for n in t["notes"]:
                    m = store.create(
                        Memory(
                            content=n["text"],
                            agent=n["agent"],
                            scope=MemoryScope.SHARED,
                            task_id=t["task_id"],
                            confidence=t["priors"][n["agent"]],
                            importance=0.5 if n["label"] is None else 0.8,
                            timestamp=_when(t, n["minutes"]),
                            claim_key=n["claim_key"],
                            claim_value=n["value"],
                            valid_until=None if n["valid_until_minutes"] is None else _when(t, n["valid_until_minutes"]),
                        )
                    )
                    labels[str(m.id)] = n
                    memories.setdefault(t["task_id"], []).append(m)
                backend.sync_from_repository(task_id=t["task_id"])

            for episode in _episodes(tasks).values():
                rel = make_reliability(rel_kind, episode[0]["priors"], params)
                trust_ms = [0.0]

                def trust_fn(m: Memory, rel=rel, acc=trust_ms) -> float:
                    start = time.perf_counter()
                    value = rel.trust(m.agent, topic_of(m.claim_key))
                    acc[0] += (time.perf_counter() - start) * 1000
                    return value

                for t in episode:
                    trust_ms[0] = 0.0
                    claim_notes = [n for n in t["notes"] if n["claim_key"] == t["claim_key"]]
                    writers = sorted({n["agent"] for n in claim_notes})
                    trust_now = {a: (rel.trust(a, t["topic"]) if rel else t["priors"][a]) for a in writers}
                    ctx = RetrievalContext(
                        agent_id="reader",
                        task_id=t["task_id"],
                        now=_when(t, t["query_minutes"]),
                        trust=trust_fn if rel else None,
                        evidence=Evidence(_events(t, "evidence_before", EVIDENCE_BEFORE_MIN)) if use_evidence else None,
                    )
                    start = time.perf_counter()
                    result = backend.retrieve(t["query"], ctx, top_k=5)
                    latency = (time.perf_counter() - start) * 1000
                    ranked = [labels[h["memory_id"]] for h in result.hits if h["memory_id"] in labels]
                    rows.append(score_task(t, condition, ranked, trust_now, writers, claim_notes,
                                           getattr(result, "contested", False), latency, trust_ms[0]))
                    if rel is not None:
                        for event in _events(t, "verify_after", VERIFY_AFTER_MIN):
                            learn(rel, event, memories[t["task_id"]])
            backend.close()
    finally:
        reset_storage_engine()
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def score_task(t, condition, ranked, trust_now, writers, claim_notes, contested, latency, trust_ms) -> dict[str, Any]:
    top = ranked[0] if ranked else None
    if top is None or top["claim_key"] != t["claim_key"]:
        outcome = "miss"
    else:
        outcome = "correct" if top["value"] == t["expected"] else "wrong"
    on_claim = [n for n in ranked if n["claim_key"] == t["claim_key"]]
    right = [i for i, n in enumerate(on_claim) if n["value"] == t["expected"]]
    wrong = [i for i, n in enumerate(on_claim) if n["value"] != t["expected"]]
    pairs = len(right) * len(wrong)
    inversions = sum(1 for w in wrong for r in right if w < r)
    authority = max(writers, key=lambda a: t["priors"][a])
    latest = {}
    for n in sorted(claim_notes, key=lambda n: n["minutes"]):
        latest[n["agent"]] = n["value"] == t["expected"]
    return {
        "task_id": t["task_id"],
        "scenario": t["scenario"],
        "seed": t["seed"],
        "index": t["index"],
        "condition": condition,
        "outcome": outcome,
        "top_agent": top["agent"] if top else None,
        "top_value": top["value"] if top else None,
        "stale": outcome == "wrong" and top["value"] in t["stale_values"],
        "wrong_authority": outcome == "wrong" and top["agent"] == authority,
        "inversion_rate": inversions / pairs if pairs else 0.0,
        "contested": bool(contested),
        "trust": trust_now,
        "brier": sum((trust_now[a] - latest[a]) ** 2 for a in writers) / len(writers),
        "latency_ms": latency,
        "trust_ms": trust_ms,
        "ranked": [{"agent": n["agent"], "value": n["value"], "label": n["label"]} for n in ranked],
    }


def summarize(rows: list[dict[str, Any]], burn_in: int) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["condition"])].append(r)
    for (sc, cond), rs in sorted(groups.items()):
        late = [r for r in rs if r["index"] >= burn_in]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        out[sc][cond] = {
            "n": len(rs),
            "correct": mean([r["outcome"] == "correct" for r in rs]),
            "correct_after_burn_in": mean([r["outcome"] == "correct" for r in late]),
            "n_after_burn_in": len(late),
            "stale": mean([r["stale"] for r in rs]),
            "wrong_authority": mean([r["wrong_authority"] for r in rs]),
            "inversion_rate": mean([r["inversion_rate"] for r in rs]),
            "contested_rate": mean([r["contested"] for r in rs]),
            "brier": mean([r["brier"] for r in rs]) if CONDITIONS[cond][0] == "collaborative" and cond != "dense_rerank" else None,
            "latency_ms": mean([r["latency_ms"] for r in rs]),
            "trust_ms": mean([r["trust_ms"] for r in rs]),
        }
    return out


def check_predictions(rows: list[dict[str, Any]], preds: dict[str, Any], burn_in: int) -> list[dict[str, Any]]:
    by = {(r["condition"], r["task_id"]): r for r in rows}
    conds = {r["condition"] for r in rows}

    def subset(sc: str, cond: str, start: int) -> list[dict[str, Any]]:
        return [r for r in rows if r["scenario"] == sc and r["condition"] == cond and r["index"] >= start]

    checks = []
    scenarios = {r["scenario"] for r in rows}
    for sc, per_cond in preds["scenarios"].items():
        for cond, metrics in per_cond.items():
            if cond not in conds or sc not in scenarios:
                continue
            for metric, want in metrics.items():
                if metric == "same_as":
                    rs = subset(sc, cond, 0)
                    diff = sum(r["outcome"] != by[(want, r["task_id"])]["outcome"] for r in rs if (want, r["task_id"]) in by)
                    checks.append({"scenario": sc, "condition": cond, "metric": f"same_as {want}",
                                   "predicted": 0, "observed": diff, "hit": diff == 0})
                    continue
                if metric == "contested_rate":
                    rs = subset(sc, cond, 0)
                    value = sum(r["contested"] for r in rs) / len(rs)
                else:
                    start = burn_in if metric == "correct_after_burn_in" else int(metric.rsplit("_", 1)[1])
                    rs = subset(sc, cond, start)
                    value = sum(r["outcome"] == "correct" for r in rs) / len(rs)
                checks.append({"scenario": sc, "condition": cond, "metric": metric, "predicted": want,
                               "observed": round(value, 3), "hit": want[0] <= value <= want[1]})
    for cond in preds["global"]["cold_start_same_as_static"]:
        if cond not in conds or "static" not in conds:
            continue
        first = [r for r in rows if r["condition"] == cond and r["index"] == 0]
        diff = sum(r["outcome"] != by[("static", r["task_id"])]["outcome"] for r in first)
        checks.append({"scenario": "all (task 0)", "condition": cond, "metric": "same_as static",
                       "predicted": 0, "observed": diff, "hit": diff == 0})
    return checks


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out_dir: Path, conditions: list[str], scenarios: list[str] | None = None) -> dict[str, Any]:
    import os

    params = load_yaml(PARAMS_PATH)
    tasks = load_tasks()
    if scenarios:
        tasks = [t for t in tasks if t["scenario"] in scenarios]
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        start = time.perf_counter()
        rows += run_condition(cond, tasks, params)
        print(f"{cond:<18} {time.perf_counter() - start:6.1f}s")
    summary = summarize(rows, params["burn_in"])
    checks = check_predictions(rows, load_yaml(PREDICTIONS_PATH), params["burn_in"])
    from utils.config import load_config

    cfg = load_config()
    offline = os.getenv("MEMORY_OFFLINE_RETRIEVAL", "").lower() in ("1", "true", "yes")
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(),
        "embeddings": "hash (offline)" if offline else cfg.embedding_model,
        "reranker": "none (offline)" if offline else cfg.reranker_model,
        "conditions": conditions,
        "n_tasks": len(tasks),
        "inputs_sha256": {p.name: _sha(p) for p in (PARAMS_PATH, SCENARIOS_PATH, TASKS_PATH, PREDICTIONS_PATH)},
        "llm_calls": 0,
        "api_cost_usd": 0.0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "predictions.json").write_text(json.dumps(checks, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "report.md").write_text(report(summary, checks, conditions))
    return {"summary": summary, "checks": checks}


def report(summary: dict[str, Any], checks: list[dict[str, Any]], conditions: list[str]) -> str:
    lines = ["# Benchmark C results", "", "Correct top-1 after burn-in (all tasks in brackets).", ""]
    lines.append("| Scenario | " + " | ".join(conditions) + " |")
    lines.append("|---|" + "---|" * len(conditions))
    for sc, per in summary.items():
        cells = [f"{per[c]['correct_after_burn_in']:.2f} ({per[c]['correct']:.2f})" if c in per else "" for c in conditions]
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    for metric, fmt in (("contested_rate", "{:.2f}"), ("stale", "{:.2f}"), ("wrong_authority", "{:.2f}"),
                        ("inversion_rate", "{:.2f}"), ("brier", "{:.3f}"), ("latency_ms", "{:.1f}"), ("trust_ms", "{:.3f}")):
        lines += ["", f"## {metric}", "", "| Scenario | " + " | ".join(conditions) + " |", "|---|" + "---|" * len(conditions)]
        for sc, per in summary.items():
            cells = ["" if c not in per or per[c][metric] is None else fmt.format(per[c][metric]) for c in conditions]
            lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    hits = sum(c["hit"] for c in checks)
    lines += ["", f"## Predictions: {hits} of {len(checks)} hit", "", "| Scenario | Condition | Metric | Predicted | Observed | |", "|---|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['scenario']} | {c['condition']} | {c['metric']} | {c['predicted']} | {c['observed']} | {'HIT' if c['hit'] else 'MISS'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tasks", "run"])
    parser.add_argument("--condition", action="append", choices=list(CONDITIONS))
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.command == "tasks":
        path = write_tasks()
        print(f"{sum(1 for _ in path.open())} tasks -> {path}")
        return 0
    out = run(args.out, args.condition or list(CONDITIONS), args.scenario)
    hits = sum(c["hit"] for c in out["checks"])
    print(f"predictions: {hits} of {len(out['checks'])} hit; report -> {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
