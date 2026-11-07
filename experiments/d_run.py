"""Benchmark D: real LLM agents, deterministic ground truth, four retrieval methods.

    python -m experiments.d_run tasks --split pilot            # freeze the controlled tasks (no LLM)
    python -m experiments.d_run estimate --split pilot         # offline call/token/cost estimate
    python -m experiments.d_run generate --split pilot --model grok-4.7 --mock --run-id mock-pilot
    python -m experiments.d_run generate --split pilot --model grok-4.7      # LIVE: needs approval
    python -m experiments.d_run evaluate --run-id pilot-v1     # offline replay + metrics (rescoring)
    python -m experiments.d_run generate --split stress --model gpt-6-luna --run-id stress-v1  # LIVE: needs approval
    python -m experiments.d_run evaluate --split stress --run-id stress-v1  # adaptive-trust stress (own tree)

Layers and where each lives:
  A. controlled environment (generated, no LLM)       d_env.py   -> tasks/<split>.jsonl
  B. real LLM agents (the only live API calls)        d_agents.py -> runs/<id>/<model>/agent_calls.jsonl, writes.jsonl
  C. deterministic ground truth (harness checks)      d_env.check -> runs/<id>/<model>/checks.jsonl

Option 1 protocol: agents run ONCE per task and model (what they see depends only on the task
script and earlier notes of the same task, never on a retrieval method). The recorded run is then
replayed through every method, each with its own memory database and trust state per model.
Per task, in order: notes are inserted -> the query is scored with trust as it is now -> only then
does the post-query external check update learned trust. Retrieval never updates trust.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from experiments import d_agents, d_env
from experiments.d_env import POSTCHECK_MIN, QUERY_MIN, at, step_minute, truth_at

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments" / "d_config.yaml"
RESULTS = ROOT / "results" / "benchmark_d"
RUNS = RESULTS / "runs"
SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|xai-[A-Za-z0-9_-]{20,})")


def load_config(split: str | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    return stress_config(cfg) if split == "stress" else cfg


def stress_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The adaptive-trust stress experiment: same models, retrieval and trust parameters; its own
    methods (plus the offline no-per-topic diagnostic), comparisons and $5 cap."""
    s = cfg["stress"]
    return {**cfg, "methods": s["methods"], "primary_comparisons": s["comparisons"], "budget_usd": {"stress": s["budget_usd"]}}


def runs_for(split: str | None) -> Path:
    """Stress traces live in their own tree, never next to pilot-v2."""
    return d_env.STRESS_DIR / "runs" if split == "stress" else RUNS


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


# --------------------------------------------------------------------------------------------
# Generate: the real agents act once per task (the only phase that can call an API)
# --------------------------------------------------------------------------------------------


def make_client(model: dict[str, Any], mock: bool, cfg: dict[str, Any]) -> d_agents.Client:
    if mock:
        return d_agents.MockClient(model["model_id"])
    if not model.get("price_verified"):
        raise SystemExit(f"{model['model_id']}: price not verified against the official pricing page; "
                         "set price_verified and price_checked_on in d_config.yaml first.")
    client = d_agents.OpenAICompatClient(model["provider"], model["model_id"],
                                         d_agents.load_api_key(model["provider"], cfg["credentials_file"]))
    if d_agents.endpoint_multiplier(client.base_url) != model.get("price_multiplier", 1.0):
        raise SystemExit(f"{model['model_id']}: price_multiplier does not match the endpoint {client.base_url}")
    return client


def generate(split: str, model_name: str, run_id: str, *, client: d_agents.Client | None = None,
             mock: bool = False, cfg: dict[str, Any] | None = None, runs_dir: Path = RUNS,
             tasks: list[dict[str, Any]] | None = None, limit: int | None = None) -> Path:
    cfg = cfg or load_config()
    if mock != run_id.startswith("mock"):
        raise SystemExit("mock runs must use a run id starting with 'mock', live runs must not")
    model = cfg["models"][model_name]
    client = client or make_client(model, mock, cfg)
    tasks = tasks if tasks is not None else d_env.load_tasks(split)
    run_dir = runs_dir / run_id
    model_dir = run_dir / model_name
    ledger = d_agents.Ledger.open(run_dir, model_dir, cfg["budget_usd"][split])
    lock = (model_dir / ".generate.lock").open("w")
    try:  # two generate processes on one model dir would record tasks twice
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another generate process is running for {model_dir}") from None
    done = {c["task_id"] for c in _jsonl(model_dir / "checks.jsonl") if c["phase"] == "after_query"}
    for t in tasks[:limit]:
        if t["task_id"] in done:
            continue
        writes, checks, shown = [], [], []
        views = iter(t["views"])
        for i, step in enumerate(t["steps"]):
            minute = step_minute(i)
            if step == "CHANGE":
                continue
            if step == "CHECK":
                checks.append(_check(t, minute, "before_query"))
                continue
            view = next(views)
            doc = d_env.view_document(t, view, minute)
            notes = [] if t.get("independent") else shown  # stress tasks: agents never see teammates' notes
            messages = d_agents.build_messages(step, t, at(t, minute).isoformat().replace("+00:00", "Z"), doc, notes)
            parsed, error, call_ids = None, None, []
            for attempt in range(cfg["calls"]["max_parse_attempts"]):
                meta = {"run_id": run_id, "model_name": model_name, "task_id": t["task_id"], "step": i,
                        "role": step, "parse_attempt": attempt}
                reply, record = ledger.call(client, model, messages, meta, cfg["calls"]["max_api_retries"])
                call_ids.append(record["timestamp"])
                try:
                    parsed = d_agents.parse_reply(reply.text, t["topic"])
                    break
                except d_agents.ReplyError as exc:
                    error = str(exc)
            truth = truth_at(t, minute)
            w = {"origin": "llm_agent_write", "run_id": run_id, "model_name": model_name, "task_id": t["task_id"],
                 "step": i, "minute": minute, "at": at(t, minute).isoformat(), "role": step, "view": view,
                 "prompt_sha256": d_agents.messages_sha256(messages), "calls": call_ids,
                 "parse_ok": parsed is not None, "parse_error": None if parsed else error,
                 "truth_at_write": truth}
            if parsed:
                w.update(parsed)
                w["key_ok"] = parsed["claim_key"] == t["claim_key"]
                w["correct_at_write"] = w["key_ok"] and parsed["claim_value"] == truth
                w["note_mentions_value"] = parsed["claim_value"] is not None and \
                    d_env._alnum(parsed["claim_value"]) in d_env._alnum(parsed["note"].replace(",", ""))
                shown.append({"at": at(t, minute).isoformat().replace("+00:00", "Z"), "role": step, "note": parsed["note"]})
            writes.append(w)
        checks.append(_check(t, POSTCHECK_MIN, "after_query"))
        _append(model_dir / "writes.jsonl", writes)
        _append(model_dir / "checks.jsonl", checks)  # the after_query check marks the task done
        print(f"[{model_name}] {t['task_id']}  spent ${ledger.spent:.4f}")
    return model_dir


def _check(t: dict[str, Any], minute: int, phase: str) -> dict[str, Any]:
    """A harness check: read the environment file and record what it says. Never an LLM."""
    return {"origin": "harness_check", "task_id": t["task_id"], "claim_key": t["claim_key"],
            "value": truth_at(t, minute), "source_kind": t["source_kind"], "source_id": f"harness:{d_env.PATHS[t['topic']]}",
            "minute": minute, "at": at(t, minute).isoformat(), "phase": phase}


# --------------------------------------------------------------------------------------------
# Evaluate: offline replay of the recorded writes through every method
# --------------------------------------------------------------------------------------------


def pinned_config(name: str, tmp: Path, cfg: dict[str, Any]):
    """Benchmark C's retrieval settings, pinned: no .env, .env.local or MEMORY_* variable can change them.
    Only offline_retrieval (hash embeddings in tests) still follows the environment."""
    from benchmarks._helpers import make_bench_config

    config = make_bench_config(name, faiss_dir=tmp).model_copy(update=cfg["retrieval_pinned"])
    assert all(getattr(config, k) == v for k, v in cfg["retrieval_pinned"].items())
    return config


def make_reliability(kind: str | None, cfg: dict[str, Any], per_topic: bool = True):
    from memory.reliability import BetaReliability

    if kind is None:
        return None
    return BetaReliability(cfg["role_prior"], prior_strength=cfg["prior_strength"], retention=cfg["retention"],
                           topic_threshold=cfg["topic_threshold"], per_topic=per_topic)


def _event(c: dict[str, Any]):
    from memory.evidence import VerificationEvent

    return VerificationEvent(claim_key=c["claim_key"], value=c["value"], source_kind=c["source_kind"],
                             source_id=c["source_id"], at=datetime.fromisoformat(c["at"]))


def replay(method: str, tasks: list[dict[str, Any]], writes: list[dict[str, Any]], checks: list[dict[str, Any]],
           cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One method over one model's recorded run, with its own fresh database and trust state."""
    from baselines.factory import create_backend
    from baselines.runtime import configure_torch_safety
    from benchmarks._helpers import reset_storage_engine
    from experiments.real_run import _ablation
    from memory.evidence import Evidence, expired, topic_of
    from memory.reliability import judge, learn
    from memory.schema import Memory, MemoryScope
    from retrieval.context import RetrievalContext
    from storage import init_db, session_scope
    from storage.repository import MemoryRepository
    from storage.sqlite_store import SQLiteMemoryStore

    spec = cfg["methods"][method]
    prior = cfg["role_prior"]
    by_task_w: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for w in writes:
        by_task_w[w["task_id"]].append(w)
    by_task_c: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in checks:
        by_task_c[c["task_id"]].append(c)
    configure_torch_safety()
    reset_storage_engine()
    tmp = Path(tempfile.mkdtemp(prefix=f"bench_d_{method}_"))
    rows, snapshots = [], []
    try:
        config = pinned_config(f"bench_d_{method}", tmp, cfg)
        init_db(config)
        with session_scope(config) as session:
            store = SQLiteMemoryStore(session)
            backend = create_backend(spec["technique"], MemoryRepository(session), config, work_dir=tmp,
                                     ablation=_ablation(spec["ablation"]))
            rel = make_reliability(spec["reliability"], cfg, spec.get("per_topic", True))
            for t in tasks:
                if not any(c["phase"] == "after_query" for c in by_task_c[t["task_id"]]):
                    continue  # task not generated (yet)
                info: dict[str, dict[str, Any]] = {}
                mems: list[Memory] = []
                for w in by_task_w[t["task_id"]]:
                    if not w["parse_ok"]:
                        continue
                    common = {"agent": w["role"], "scope": MemoryScope.SHARED, "task_id": t["task_id"],
                              "confidence": prior[w["role"]], "timestamp": datetime.fromisoformat(w["at"])}
                    m = store.create(Memory(content=w["note"], importance=0.8, claim_key=w["claim_key"],
                                            claim_value=w["claim_value"],
                                            valid_until=datetime.fromisoformat(w["valid_until"]) if w["valid_until"] else None,
                                            **common))
                    info[str(m.id)] = {**w, "kind": "note"}
                    mems.append(m)
                    if w["other_note"]:
                        o = store.create(Memory(content=w["other_note"], importance=0.5, **common))
                        info[str(o.id)] = {**w, "kind": "other", "claim_key": None, "claim_value": None}
                backend.sync_from_repository(task_id=t["task_id"])
                before = [_event(c) for c in by_task_c[t["task_id"]] if c["phase"] == "before_query"]
                now = at(t, QUERY_MIN)
                trust_now = {r: (rel.trust(r, t["topic"]) if rel else prior[r]) for r in prior}
                ctx = RetrievalContext(
                    agent_id="reader", task_id=t["task_id"], now=now,
                    # Unkeyed notes (the "other_note") take the task's topic, so an agent's notes in one
                    # task share one trust value, as they share one confidence under static trust.
                    trust=(lambda m, rel=rel, tp=t["topic"]: rel.trust(m.agent, topic_of(m.claim_key) or tp)) if rel else None,
                    evidence=Evidence(before) if spec["evidence"] else None,
                )
                start = time.perf_counter()
                result = backend.retrieve(t["query"], ctx, top_k=cfg["top_k"])
                latency = (time.perf_counter() - start) * 1000
                ranked = [info[h["memory_id"]] for h in result.hits if h["memory_id"] in info]
                live = [m for m in mems if m.claim_key == t["claim_key"] and not expired(m, now)]
                rows.append(score(t, method, ranked, live, before, result.contested, latency, trust_now, prior,
                                  [info[str(m.id)] for m in mems]))
                if rel is not None:
                    judged = []
                    for c in by_task_c[t["task_id"]]:
                        if c["phase"] == "after_query":
                            judged += judge(_event(c), mems)
                            learn(rel, _event(c), mems)
                    snapshots.append({"origin": "replay", "method": method, "task_id": t["task_id"], "topic": t["topic"],
                                      "judged": [[a, ok] for a, ok in judged],
                                      "after_learning": {f"{a}|{tp}": round(rel.trust(a, tp), 6)
                                                         for a in prior for tp in [None, *d_env.TOPICS]}})
            backend.close()
    finally:
        reset_storage_engine()
        shutil.rmtree(tmp, ignore_errors=True)
    return rows, snapshots


def score(t, method, ranked, live, before, contested, latency, trust_now, prior, task_notes) -> dict[str, Any]:
    from memory.evidence import Evidence

    verdict = Evidence(before).verdict(t["claim_key"]) if before else None

    def label(n: dict[str, Any]) -> str:
        if n["kind"] != "note" or n["claim_key"] != t["claim_key"]:
            return "off_key"
        if n["claim_value"] == t["truth"]:
            return "correct"
        if verdict and verdict.value is not None and n["claim_value"] != verdict.value and \
                datetime.fromisoformat(n["at"]) <= verdict.at:
            return "refuted"
        return "stale" if n["claim_value"] in t["stale_values"] else "wrong"

    labels = [label(n) for n in ranked]
    latest: dict[str, Any] = {}
    for m in sorted(live, key=lambda m: m.timestamp):
        latest[m.agent] = m.claim_value
    values = set(latest.values())
    right = [a for a, v in latest.items() if v == t["truth"]]
    wrong = [a for a, v in latest.items() if v != t["truth"]]
    lower_case = bool(right and wrong and max(prior[a] for a in right) < max(prior[a] for a in wrong))
    return {
        "origin": "replay", "task_id": t["task_id"], "scenario": t["scenario"], "topic": t["topic"],
        "position": t["position"], "method": method, "truth": t["truth"],
        "phase": t.get("phase"), "lower_agent": t.get("lower_agent"),
        "correct@1": labels[:1] == ["correct"], "correct@3": "correct" in labels[:3],
        "stale@1": labels[:1] == ["stale"], "refuted@1": labels[:1] == ["refuted"], "off_key@1": labels[:1] in (["off_key"], []),
        "has_pre_query_evidence": bool(before),
        "oracle": any(n["kind"] == "note" and n["claim_key"] == t["claim_key"] and n["claim_value"] == t["truth"]
                      for n in task_notes),
        "conflict": len(values) > 1, "contested": bool(contested),
        "lower_prior_case": lower_case,
        "top_role": ranked[0]["role"] if ranked else None,
        "trust_at_query": {a: round(v, 6) for a, v in trust_now.items()},
        "latency_ms": round(latency, 3),
        "ranked": [{"role": n["role"], "kind": n["kind"], "claim_value": n["claim_value"], "label": lb, "step": n["step"]}
                   for n, lb in zip(ranked, labels, strict=True)],
    }


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def prop(rows: list[dict[str, Any]], metric: str, where: str | None = None) -> dict[str, Any]:
    rs = [r for r in rows if where is None or r[where]]
    k = sum(bool(r[metric]) for r in rs)
    return {"k": k, "n": len(rs), "rate": round(k / len(rs), 4) if rs else None, "ci95": wilson(k, len(rs))}


def paired_bootstrap(rows: list[dict[str, Any]], a: str, b: str, resamples: int, seed: int) -> dict[str, Any]:
    """Difference in correct@1 (a - b), resampling tasks (all models of a task move together)."""
    by: dict[str, dict[tuple[str, str], bool]] = defaultdict(dict)
    for r in rows:
        if r["method"] in (a, b):
            by[r["task_id"]][(r["model_name"], r["method"])] = r["correct@1"]
    units = sorted(tid for tid, v in by.items() if {m for _, m in v} == {a, b})
    if not units:
        return {"n_tasks": 0}

    def diff(ids: list[str]) -> float:
        da = [v for tid in ids for (mn, m), v in by[tid].items() if m == a]
        db = [v for tid in ids for (mn, m), v in by[tid].items() if m == b]
        return sum(da) / len(da) - sum(db) / len(db)

    rng = random.Random(seed)
    sims = sorted(diff([rng.choice(units) for _ in units]) for _ in range(resamples))
    return {"a": a, "b": b, "n_tasks": len(units), "diff": round(diff(units), 4),
            "ci95": (round(sims[int(0.025 * resamples)], 4), round(sims[int(0.975 * resamples) - 1], 4))}


METRICS = ["correct@1", "correct@3", "stale@1", "off_key@1", "oracle"]


def summarize(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[f"{r['model_name']}|{r['method']}"].append(r)
        groups[f"ALL|{r['method']}"].append(r)
        groups[f"{r['model_name']}|{r['method']}|{r['scenario']}"].append(r)
        groups[f"ALL|{r['method']}|{r['scenario']}"].append(r)
    out: dict[str, Any] = {}
    for g, rs in sorted(groups.items()):
        s = {m: prop(rs, m) for m in METRICS}
        s["refuted@1 (tasks with pre-query evidence)"] = prop(rs, "refuted@1", "has_pre_query_evidence")
        s["correct@1 (lower-prior agent right)"] = prop(rs, "correct@1", "lower_prior_case")
        s["conflict recall"] = prop(rs, "contested", "conflict")
        s["conflict precision"] = prop([{**r, "both": r["conflict"]} for r in rs], "both", "contested")
        s["latency_ms_mean"] = round(sum(r["latency_ms"] for r in rs) / len(rs), 3)
        out[g] = s
    comps = {}
    for a, b in cfg["primary_comparisons"]:
        comps[f"{a} - {b} (pooled)"] = paired_bootstrap(rows, a, b, cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"])
        for mn in sorted({r["model_name"] for r in rows}):
            comps[f"{a} - {b} ({mn})"] = paired_bootstrap([r for r in rows if r["model_name"] == mn], a, b,
                                                         cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"])
    return {"groups": out, "comparisons": comps}


def agent_summary(model_dir: Path) -> dict[str, Any]:
    calls = [c for c in _jsonl(model_dir / "agent_calls.jsonl") if "usage" in c]
    writes = _jsonl(model_dir / "writes.jsonl")
    tok = lambda f: sum(c["usage"].get(f, 0) for c in calls)  # noqa: E731
    lat = sorted(c["latency_ms"] for c in calls)
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for w in writes:
        by_view[f"{w['role']}|{w['view']}"].append(w)
    ok = [w for w in writes if w["parse_ok"]]
    return {
        "calls": len(calls), "api_errors": sum(1 for c in _jsonl(model_dir / "agent_calls.jsonl") if "error" in c),
        "models_returned": sorted({c["model_returned"] for c in calls}),
        "truncated_by_token_cap": sum(c.get("finish_reason") == "length" for c in calls),
        "prompt_tokens": tok("prompt_tokens"), "completion_tokens": tok("completion_tokens"),
        "reasoning_tokens": tok("reasoning_tokens"), "cached_tokens": tok("cached_tokens"),
        "cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
        "latency_ms_p50": lat[len(lat) // 2] if lat else None, "latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else None,
        "writes": len(writes),
        "parse_failed": prop([{"x": not w["parse_ok"]} for w in writes], "x"),
        "wrong_or_invalid_key": prop([{"x": not w["key_ok"]} for w in ok], "x"),
        "value_unreadable": prop([{"x": w["claim_value"] is None} for w in ok], "x"),
        "note_mentions_value": prop(ok, "note_mentions_value"),
        "expiry_recorded": prop([{"x": w["valid_until"] is not None} for w in ok], "x"),
        "correct_at_write_by_role_view": {k: prop([{"x": w.get("correct_at_write", False)} for w in v], "x")
                                          for k, v in sorted(by_view.items())},
    }


def assert_no_secrets(run_dir: Path, credentials_file: str | None = None) -> None:
    secrets = [v for v in (os.getenv("XAI_API_KEY"), os.getenv("OPENAI_API_KEY")) if v and len(v) >= 8]
    if credentials_file:
        for provider in d_agents.PROVIDERS:
            try:
                secrets.append(d_agents.load_api_key(provider, credentials_file))  # compared, never shown
            except SystemExit:
                pass
    for p in run_dir.rglob("*"):
        if p.is_file():
            text = p.read_text(errors="ignore")
            if SECRET_PATTERN.search(text) or any(s in text for s in secrets):
                raise SystemExit(f"possible credential in {p.relative_to(run_dir)}; refusing to write checksums")


def dedupe(writes: list[dict[str, Any]], checks: list[dict[str, Any]]) -> tuple[list, list, dict[str, int]]:
    """Keep the first record per (task, step) and per (task, check phase, minute). Duplicates can
    only come from overlapping generate processes; the raw traces keep them, scoring does not."""
    seen_w, seen_c, w, c = set(), set(), [], []
    for x in writes:
        if (k := (x["task_id"], x["step"])) not in seen_w:
            seen_w.add(k)
            w.append(x)
    for x in checks:
        if (k := (x["task_id"], x["phase"], x["minute"])) not in seen_c:
            seen_c.add(k)
            c.append(x)
    return w, c, {"writes_dropped": len(writes) - len(w), "checks_dropped": len(checks) - len(c)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(run_id: str, *, eval_name: str = "eval-v1", runs_dir: Path = RUNS, cfg: dict[str, Any] | None = None,
             split: str | None = None, tasks: list[dict[str, Any]] | None = None, overwrite: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    run_dir = runs_dir / run_id
    out = run_dir / eval_name
    if out.exists() and not overwrite:
        raise SystemExit(f"{out} exists; use a new --eval-name (old artifacts are kept)")
    model_dirs = sorted(p for p in run_dir.iterdir() if (p / "writes.jsonl").exists())
    if not model_dirs:
        raise SystemExit(f"no generated model runs in {run_dir}")
    if tasks is None:
        split = split or _jsonl(model_dirs[0] / "writes.jsonl")[0]["task_id"].split("-", 1)[0]
        tasks = d_env.load_tasks(split)
    rows: list[dict[str, Any]] = []
    out.mkdir(parents=True, exist_ok=True)
    agents, snaps_by, duplicates = {}, {}, {}
    for md in model_dirs:
        writes, checks, duplicates[md.name] = dedupe(_jsonl(md / "writes.jsonl"), _jsonl(md / "checks.jsonl"))
        (out / md.name).mkdir(exist_ok=True)
        for f in ("scores.jsonl", "trust.jsonl"):
            (out / md.name / f).unlink(missing_ok=True)
        for method in cfg["methods"]:
            r, snaps = replay(method, tasks, writes, checks, cfg)
            for x in r:
                x["model_name"] = md.name
            _append(out / md.name / "scores.jsonl", r)
            _append(out / md.name / "trust.jsonl", snaps)
            snaps_by[(md.name, method)] = snaps
            rows += r
        agents[md.name] = agent_summary(md)
    summary = {"retrieval": summarize(rows, cfg), "agents": agents}
    if split == "stress":
        summary["stress"] = {md.name: stress_analysis(tasks, [r for r in rows if r["model_name"] == md.name],
                                                      {m: snaps_by[(md.name, m)] for m in cfg["methods"]},
                                                      dedupe(_jsonl(md / "writes.jsonl"), [])[0], cfg["role_prior"])
                             for md in model_dirs}
    offline = os.getenv("MEMORY_OFFLINE_RETRIEVAL", "").lower() in ("1", "true", "yes")
    manifest = {
        "run_id": run_id, "eval_name": eval_name, "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(),
        "embeddings": "hash (offline)" if offline else cfg["retrieval_pinned"]["embedding_model"],
        "reranker": "none (offline)" if offline else cfg["retrieval_pinned"]["reranker_model"],
        "retrieval_settings": {**cfg["retrieval_pinned"], "offline_retrieval": offline},
        "models": {md.name: agents[md.name]["models_returned"] for md in model_dirs},
        "duplicate_records_dropped": duplicates,
        "n_tasks_scored": {md.name: len({r["task_id"] for r in rows if r["model_name"] == md.name}) for md in model_dirs},
        "llm_calls": sum(a["calls"] for a in agents.values()),
        "api_cost_usd": round(sum(a["cost_usd"] for a in agents.values()), 6),
        "prompt_templates_sha256": d_agents.prompt_templates_sha256(),
        "inputs_sha256": {p.name: _sha(p) for p in (CONFIG_PATH, ROOT / "experiments" / "d_env.py",
                                                     ROOT / "experiments" / "d_agents.py", Path(__file__))},
        "tasks_sha256": _sha(d_env.tasks_path(split)) if split and d_env.tasks_path(split).exists() else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out / "report.md").write_text(report(summary, cfg) + (stress_report(summary["stress"]) if "stress" in summary else ""))
    assert_no_secrets(run_dir, cfg.get("credentials_file"))
    sums = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and not p.name.startswith("."))
    (run_dir / "SHA256SUMS").write_text("".join(f"{_sha(p)}  {p.relative_to(run_dir)}\n" for p in sums))
    return summary


# --------------------------------------------------------------------------------------------
# Adaptive-trust stress analysis: when trust changes vs when rankings change (separate outcomes)
# --------------------------------------------------------------------------------------------

DEGRADED, CONTROL = d_env.STRESS_DEGRADED, d_env.STRESS_CONTROL


def stress_analysis(tasks: list[dict[str, Any]], rows: list[dict[str, Any]], snaps: dict[str, list[dict[str, Any]]],
                    writes: list[dict[str, Any]], prior: dict[str, float]) -> dict[str, Any]:
    """One model's stream. Every count of 'incorrect verifier checks' is the number of post-query
    harness checks, in earlier tasks, that judged the verifier's deps claim wrong (from the
    recorded judgements, not from the scripted view)."""
    order = [t["task_id"] for t in tasks]
    by = {(r["method"], r["task_id"]): r for r in rows}
    judged = {s["task_id"]: s["judged"] for s in snaps.get("learned", [])}
    wrong_before, n = {}, 0
    for t in tasks:
        wrong_before[t["task_id"]] = n
        n += sum(1 for a, ok in judged.get(t["task_id"], []) if a == "verifier" and not ok and t["topic"] == DEGRADED)
    deps2 = [t for t in tasks if t["topic"] == DEGRADED and t.get("phase") == 2 and ("learned", t["task_id"]) in by]

    def first(pred, method: str = "learned") -> dict[str, Any] | None:
        for t in deps2:
            if pred(t, by[(method, t["task_id"])]):
                return {"task_id": t["task_id"], "phase2_deps_index": deps2.index(t),
                        "incorrect_verifier_checks_before": wrong_before[t["task_id"]]}
        return None

    def trust_swap(method: str, role: str | None):
        def pred(t, r):
            low = role or t["lower_agent"]
            return r["trust_at_query"]["verifier"] < r["trust_at_query"][low]
        return first(pred, method)

    top = lambda m, tid: (by[(m, tid)]["ranked"][0]["role"] if by.get((m, tid)) and by[(m, tid)]["ranked"] else None)  # noqa: E731
    transitions = {}
    for m in ("learned", "learned_no_topic"):
        if not any(k[0] == m for k in by):
            continue
        transitions[m] = {
            "trust_first_drop_below_prior": first(lambda t, r: r["trust_at_query"]["verifier"] < prior["verifier"] - 1e-9, m),
            "trust_swap_vs_task_lower_agent": trust_swap(m, None),
            "trust_swap_vs_coder": trust_swap(m, "coder"),
            "trust_swap_vs_planner": trust_swap(m, "planner"),
            "ranking_first_top1_role_differs_from_static": first(lambda t, r, m=m: top(m, t["task_id"]) != top("static", t["task_id"]), m),
            "first_correct_where_static_wrong": first(
                lambda t, r: r["correct@1"] and not by[("static", t["task_id"])]["correct@1"], m),
        }

    def differ(m: str, scen: str) -> dict[str, Any]:
        ids = [t["task_id"] for t in tasks if t["scenario"] == scen and (m, t["task_id"]) in by]
        k = sum(by[(m, i)]["ranked"][:1] != by[("static", i)]["ranked"][:1] for i in ids)
        return {"k": k, "n": len(ids)}

    scens = sorted({t["scenario"] for t in tasks})
    ranking_changed = {m: {sc: differ(m, sc) for sc in scens} for m in ("learned", "learned_evidence", "learned_no_topic")
                       if any(k[0] == m for k in by)}
    control = {}
    for m in ("learned", "learned_no_topic"):
        s2 = [s for s in snaps.get(m, []) if s["task_id"] in order and tasks[order.index(s["task_id"])].get("phase") == 2]
        if s2:
            control[m] = {"min_verifier_config_trust_phase2": min(s["after_learning"][f"verifier|{CONTROL}"] for s in s2),
                          "verifier_config_trust_end": s2[-1]["after_learning"][f"verifier|{CONTROL}"],
                          "verifier_deps_trust_end": s2[-1]["after_learning"][f"verifier|{DEGRADED}"]}
    # Agent behaviour: did the scripted reliability change actually show up in the real agents?
    wt = {t["task_id"]: t for t in tasks}
    agents: dict[str, list[bool]] = defaultdict(list)
    for w in writes:
        t = wt.get(w["task_id"])
        if t:
            who = "verifier" if w["role"] == "verifier" else "lower_agent"
            agents[f"{t['scenario']}|{who}|{w['view']}"].append(bool(w.get("correct_at_write")))
    return {
        "transitions": transitions,
        "ranking_changed_vs_static_top1": ranking_changed,
        "topic_control": control,
        "agent_correct_at_write": {k: {"k": sum(v), "n": len(v)} for k, v in sorted(agents.items())},
        "verifier_wrong_deps_checks_total": n,
        "trust_trace": [{"task_id": tid, "scenario": by[("learned", tid)]["scenario"],
                         "lower_agent": by[("learned", tid)]["lower_agent"],
                         "learned": by[("learned", tid)]["trust_at_query"],
                         "learned_no_topic": by.get(("learned_no_topic", tid), {}).get("trust_at_query"),
                         "incorrect_verifier_checks_before": wrong_before[tid],
                         "top1": {m: top(m, tid) for m in ("static", "learned", "learned_no_topic") if (m, tid) in by},
                         "correct@1": {m: by[(m, tid)]["correct@1"] for m in ("dense", "static", "learned", "learned_evidence", "learned_no_topic") if (m, tid) in by}}
                        for tid in order if ("learned", tid) in by],
    }


def stress_report(stress: dict[str, Any]) -> str:
    lines = ["", "# Adaptive-trust stress analysis", "",
             "Counts in brackets: incorrect verifier checks on deps (post-query harness checks) before the transition.", ""]
    for mn, a in stress.items():
        lines += [f"## {mn}", "", "| Transition | learned | learned_no_topic |", "|---|---|---|"]
        for k in ("trust_first_drop_below_prior", "trust_swap_vs_task_lower_agent", "trust_swap_vs_coder",
                  "trust_swap_vs_planner", "ranking_first_top1_role_differs_from_static", "first_correct_where_static_wrong"):
            cell = lambda m: (lambda x: "never" if x is None else f"{x['task_id']} [{x['incorrect_verifier_checks_before']}]")(  # noqa: E731
                a["transitions"].get(m, {}).get(k)) if m in a["transitions"] else "n/a"
            lines.append(f"| {k} | {cell('learned')} | {cell('learned_no_topic')} |")
        lines += ["", "Top-1 differs from static (k/n tasks):", "", "| Scenario | " + " | ".join(a["ranking_changed_vs_static_top1"]) + " |",
                  "|---|" + "---|" * len(a["ranking_changed_vs_static_top1"])]
        for sc in next(iter(a["ranking_changed_vs_static_top1"].values()), {}):
            lines.append(f"| {sc} | " + " | ".join(f"{v[sc]['k']}/{v[sc]['n']}" for v in a["ranking_changed_vs_static_top1"].values()) + " |")
        lines += ["", f"Topic control: `{json.dumps(a['topic_control'])}`", "",
                  f"Verifier judged wrong on deps: {a['verifier_wrong_deps_checks_total']} checks", ""]
    return "\n".join(lines) + "\n"


def _fmt(p: dict[str, Any]) -> str:
    if not p or not p.get("n"):
        return "n/a (n=0)"
    lo, hi = p["ci95"]
    return f"{p['rate']:.2f} ({p['k']}/{p['n']}) [{lo:.2f}, {hi:.2f}]"


def report(summary: dict[str, Any], cfg: dict[str, Any]) -> str:
    g, methods = summary["retrieval"]["groups"], list(cfg["methods"])
    models = sorted({k.split("|")[0] for k in g} - {"ALL"}) + ["ALL"]
    lines = ["# Benchmark D results", "",
             "Rate (k/n) [Wilson 95% CI]. Sequential replay of one recorded agent run per model; ",
             "trust learns only from the post-query harness check. `oracle` = a note with the true value exists.", ""]
    for metric in [*METRICS, "refuted@1 (tasks with pre-query evidence)", "correct@1 (lower-prior agent right)",
                   "conflict recall", "conflict precision"]:
        lines += [f"## {metric}", "", "| Model | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
        for mn in models:
            lines.append(f"| {mn} | " + " | ".join(_fmt(g.get(f"{mn}|{m}", {}).get(metric, {})) for m in methods) + " |")
        lines.append("")
    scen = sorted({k.split("|")[2] for k in g if k.startswith("ALL|") and k.count("|") == 2})
    lines += ["## correct@1 by scenario, all models (descriptive only)", "",
              "| Scenario | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for sc in scen:
        lines.append(f"| {sc} | " + " | ".join(_fmt(g.get(f"ALL|{m}|{sc}", {}).get("correct@1", {})) for m in methods) + " |")
    lines += ["", "## Paired differences in correct@1 (bootstrap over tasks)", "", "| Comparison | n tasks | diff | 95% CI |", "|---|---|---|---|"]
    for name, c in summary["retrieval"]["comparisons"].items():
        lines.append(f"| {name} | {c['n_tasks']} | {c.get('diff', 'n/a')} | {c.get('ci95', 'n/a')} |")
    lines += ["", "## Agents (real LLM calls)", "", "| Model | calls | prompt tok | completion tok | reasoning tok | cost USD | p50 ms | parse failed | wrong key |",
              "|---|---|---|---|---|---|---|---|---|"]
    for mn, a in summary["agents"].items():
        lines.append(f"| {mn} | {a['calls']} | {a['prompt_tokens']} | {a['completion_tokens']} | {a['reasoning_tokens']} | "
                     f"{a['cost_usd']:.4f} | {a['latency_ms_p50']} | {_fmt(a['parse_failed'])} | {_fmt(a['wrong_or_invalid_key'])} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# Offline estimate (no API): render every prompt with the mock agent and price it
# --------------------------------------------------------------------------------------------


def estimate(split: str, cfg: dict[str, Any] | None = None, out_low: int = 300, out_high: int = 2000) -> dict[str, Any]:
    cfg = cfg or load_config()
    tasks = d_env.load_tasks(split)
    with tempfile.TemporaryDirectory() as tmp:
        name = next(iter(cfg["models"]))
        generate(split, name, "mock-estimate", mock=True, cfg=cfg, runs_dir=Path(tmp), tasks=tasks)
        calls = _jsonl(Path(tmp) / "mock-estimate" / name / "agent_calls.jsonl")
        prompts = {p["prompt_sha256"]: p for p in _jsonl(Path(tmp) / "mock-estimate" / "prompts.jsonl")}
    in_tokens = [sum(len(m["content"]) for m in prompts[c["prompt_sha256"]]["messages"]) / 4 for c in calls]
    n, mean_in = len(calls), sum(in_tokens) / len(calls)
    per_model = {}
    for mn, m in cfg["models"].items():
        lo = n * (mean_in * m["price_in"] + out_low * m["price_out"]) / 1e6
        hi = n * 1.1 * (mean_in * m["price_in"] + out_high * m["price_out"]) / 1e6  # +10% parse retries
        per_model[mn] = {"calls": n, "calls_with_retries": math.ceil(n * 1.1), "usd_low": round(lo, 3), "usd_high": round(hi, 3),
                         "price_verified": m["price_verified"]}
    return {"split": split, "tasks": len(tasks), "calls_per_model": n, "mean_input_tokens_est": round(mean_in),
            "output_tokens_assumed": [out_low, out_high], "per_model": per_model,
            "total_usd": [round(sum(p["usd_low"] for p in per_model.values()), 2),
                          round(sum(p["usd_high"] for p in per_model.values()), 2)],
            "budget_usd": cfg["budget_usd"][split]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["tasks", "estimate", "generate", "evaluate"])
    ap.add_argument("--split", choices=["pilot", "full", "stress"])
    ap.add_argument("--model")
    ap.add_argument("--run-id")
    ap.add_argument("--mock", action="store_true", help="offline mock agents (run id must start with 'mock')")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--eval-name", default="eval-v1")
    args = ap.parse_args(argv)
    cfg = load_config(args.split)
    runs_dir = runs_for(args.split)
    if args.command == "tasks" and args.split == "stress":
        s = cfg["stress"]
        tasks = d_env.generate_stress_tasks(s["seed"], s["phase1"], s["phase2"])
        print(f"{len(tasks)} tasks -> {d_env.write_tasks('stress', s['seed'], len(tasks), tasks)}")
    elif args.command == "tasks":
        s = cfg["splits"][args.split]
        print(f"{s['tasks']} tasks -> {d_env.write_tasks(args.split, s['seed'], s['tasks'])}")
    elif args.command == "estimate":
        print(json.dumps(estimate(args.split, cfg), indent=2))
    elif args.command == "generate":
        run_id = args.run_id or (f"mock-{args.split}" if args.mock else f"{args.split}-v1")
        generate(args.split, args.model, run_id, mock=args.mock, cfg=cfg, limit=args.limit, runs_dir=runs_dir)
    else:
        evaluate(args.run_id, eval_name=args.eval_name, cfg=cfg, split=args.split, runs_dir=runs_dir)
        print(f"report -> {runs_dir / args.run_id / args.eval_name / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
