"""Benchmark D, layer A and C: the CONTROLLED task environment and its deterministic ground truth.

Nothing in this module calls an LLM. It generates small projects (a config file, an API spec,
a dependency lock file, a Python module), a scripted timeline of changes to one fact per task,
and which version of which document each agent step is shown. It never decides which agent
is right: correctness is computed afterwards by reading the environment (`check`), at the
time that matters, and comparing values with `normalize`.
"""

from __future__ import annotations

import json
import random
import re
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "results" / "benchmark_d" / "tasks"
EPOCH = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)  # task i of a stream happens on day i
CHANGE_MIN = 5  # environment change before the first agent step ("changed" timeline)
EXPIRY_MIN = 100  # end of a temporary override
QUERY_MIN = 120  # the retrieval query
POSTCHECK_MIN = 130  # the independent check that trust learns from (after the query)

# --------------------------------------------------------------------------------------------
# Claim bank: key -> where the fact lives, its kind, its label in the file, values, queries.
# --------------------------------------------------------------------------------------------

CLAIMS: dict[str, dict[str, Any]] = {
    "config.db_pool_size": {"kind": "int", "label": "pool_size", "desc": "database connection pool size",
                            "values": [10, 20, 25, 40, 50, 64, 80, 100],
                            "queries": ["How many connections does the database pool allow?", "What is the db pool size?"]},
    "config.request_timeout_s": {"kind": "int", "label": "request_timeout_s", "desc": "HTTP request timeout in seconds",
                                 "values": [5, 10, 15, 30, 45, 60, 90],
                                 "queries": ["What is the HTTP request timeout?", "How long before an outgoing request times out?"]},
    "config.cache_backend": {"kind": "enum", "label": "cache_backend", "desc": "cache backend",
                             "values": ["redis", "memcached", "in-memory"],
                             "queries": ["Which cache backend does the service use?", "What are we using for caching?"]},
    "config.log_level": {"kind": "enum", "label": "log_level", "desc": "production log level",
                         "values": ["debug", "info", "warning", "error"],
                         "queries": ["What log level runs in production?", "Which logging level is configured?"]},
    "api.rate_limit_per_min": {"kind": "int", "label": "Rate limit", "desc": "API rate limit (requests per minute per key)",
                               "values": [100, 300, 500, 600, 1000, 1200, 2000],
                               "queries": ["What is the API rate limit per key?", "How many requests per minute can a client make?"]},
    "api.max_page_size": {"kind": "int", "label": "Max page size", "desc": "maximum page size of list endpoints",
                          "values": [25, 50, 100, 200, 250, 500],
                          "queries": ["What is the maximum page size for list endpoints?", "How many items can one page return?"]},
    "api.auth_scheme": {"kind": "enum", "label": "Auth scheme", "desc": "API authentication scheme",
                        "values": ["bearer", "api-key", "hmac", "oauth2"],
                        "queries": ["How do clients authenticate to the API?", "Which auth scheme does the API use?"]},
    "api.base_path_version": {"kind": "enum", "label": "Base path", "desc": "API version in the base path",
                              "values": ["v1", "v2", "v3", "v4"],
                              "queries": ["Which API version is in the base path?", "What is the current API base path version?"]},
    "deps.requests_version": {"kind": "semver", "label": "requests", "desc": "pinned version of requests",
                              "values": ["2.28.2", "2.31.0", "2.32.3", "2.33.1"],
                              "queries": ["Which requests version is pinned?", "What version of requests do we lock to?"]},
    "deps.pydantic_version": {"kind": "semver", "label": "pydantic", "desc": "pinned version of pydantic",
                              "values": ["1.10.15", "2.5.3", "2.7.4", "2.9.2", "2.11.1"],
                              "queries": ["Which pydantic version is pinned?", "What pydantic version does the lock file use?"]},
    "deps.python_version": {"kind": "semver", "label": "python_version", "desc": "Python version the project runs on",
                            "values": ["3.10", "3.11", "3.12", "3.13"],
                            "queries": ["Which Python version does the project run on?", "What Python version is required?"]},
    "deps.sqlalchemy_version": {"kind": "semver", "label": "sqlalchemy", "desc": "pinned version of SQLAlchemy",
                                "values": ["1.4.52", "2.0.25", "2.0.36", "2.1.2"],
                                "queries": ["Which SQLAlchemy version is pinned?", "What version of SQLAlchemy do we use?"]},
    "impl.retry_max_attempts": {"kind": "int", "label": "RETRY_MAX_ATTEMPTS", "desc": "maximum retry attempts of the HTTP client",
                                "values": [2, 3, 4, 5, 6, 8],
                                "queries": ["How many times does the client retry a failed request?", "What is the retry limit?"]},
    "impl.backoff_base_ms": {"kind": "int", "label": "BACKOFF_BASE_MS", "desc": "base retry backoff in milliseconds",
                             "values": [50, 100, 200, 250, 500],
                             "queries": ["What is the base retry backoff?", "How long is the first backoff delay?"]},
    "impl.hash_algorithm": {"kind": "enum", "label": "HASH_ALGORITHM", "desc": "hash algorithm used for content digests",
                            "values": ["sha256", "blake2b", "md5", "sha1"],
                            "queries": ["Which hash algorithm do content digests use?", "What do we hash payloads with?"]},
    "impl.date_format": {"kind": "enum", "label": "DATE_FORMAT", "desc": "timestamp format in API payloads",
                         "values": ["iso8601", "rfc2822", "unix"],
                         "queries": ["What timestamp format do payloads use?", "How are dates serialised?"]},
}
TOPICS = ["config", "api", "deps", "impl"]
PATHS = {"config": "config/app.toml", "api": "docs/api_spec.md", "deps": "requirements.lock", "impl": "src/client.py"}
PROJECTS = ["billing", "search", "notifications", "reporting", "auth-gateway", "ingest"]


def keys_for(topic: str) -> list[str]:
    return [k for k in CLAIMS if k.startswith(topic + ".")]


# --------------------------------------------------------------------------------------------
# Value normalisation (deterministic; used for agent claims and for ground truth alike)
# --------------------------------------------------------------------------------------------


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def normalize(key: str, raw: Any) -> str | None:
    """Canonical string for a value of `key`, or None when it can't be read unambiguously."""
    if raw is None:
        return None
    text = str(raw).strip()
    kind = CLAIMS[key]["kind"]
    if kind == "int":
        nums = {int(n.replace(",", "").replace("_", "")) for n in re.findall(r"(?<![\d.])\d[\d,_]*(?![\d.])", text)}
        return str(nums.pop()) if len(nums) == 1 else None
    if kind == "semver":
        found = {re.sub(r"(\.0)+$", "", v) for v in re.findall(r"\d+(?:\.\d+)+", text)}
        return found.pop() if len(found) == 1 else None
    hits = [v for v in CLAIMS[key]["values"] if _alnum(v) in _alnum(text)]
    hits = [h for h in hits if not any(h != o and _alnum(h) in _alnum(o) for o in hits)]
    return _alnum(hits[0]) if len(hits) == 1 else None


# --------------------------------------------------------------------------------------------
# Files: rendering and the deterministic checks that read them back
# --------------------------------------------------------------------------------------------


def render(topic: str, values: dict[str, Any], stamp: str, note: str = "") -> str:
    v = {k.split(".", 1)[1]: values[k] for k in keys_for(topic)}
    if topic == "config":
        return (f"# app.toml  (last modified {stamp})\n[database]\npool_size = {v['db_pool_size']}\n"
                f"[http]\nrequest_timeout_s = {v['request_timeout_s']}\n[cache]\ncache_backend = \"{v['cache_backend']}\"\n"
                f"[logging]\nlog_level = \"{v['log_level']}\"\n{note}")
    if topic == "api":
        return (f"# Public API spec  (last modified {stamp})\n- Base path: /api/{v['base_path_version']}\n"
                f"- Auth scheme: {v['auth_scheme']}\n- Rate limit: {v['rate_limit_per_min']} requests per minute per key\n"
                f"- Max page size: {v['max_page_size']} items\n{note}")
    if topic == "deps":
        return (f"# requirements.lock  (last modified {stamp})\n# python_version: {v['python_version']}\n"
                f"pydantic=={v['pydantic_version']}\nrequests=={v['requests_version']}\n"
                f"sqlalchemy=={v['sqlalchemy_version']}\n{note}")
    return (f"# src/client.py  (last modified {stamp})\nRETRY_MAX_ATTEMPTS = {v['retry_max_attempts']}\n"
            f"BACKOFF_BASE_MS = {v['backoff_base_ms']}\nHASH_ALGORITHM = \"{v['hash_algorithm']}\"\n"
            f"DATE_FORMAT = \"{v['date_format']}\"\n{note}")


def check(topic: str, text: str, key: str) -> str | None:
    """Read `key` from a rendered file the way a test or config read would. No LLM involved."""
    name = key.split(".", 1)[1]
    if topic == "config":
        data = tomllib.loads(text)
        section = {"db_pool_size": ("database", "pool_size"), "request_timeout_s": ("http", "request_timeout_s"),
                   "cache_backend": ("cache", "cache_backend"), "log_level": ("logging", "log_level")}[name]
        return normalize(key, data[section[0]][section[1]])
    if topic == "impl":
        ns: dict[str, Any] = {}
        exec(compile(text, PATHS["impl"], "exec"), ns)  # noqa: S102 - generated code, our own file
        return normalize(key, ns[CLAIMS[key]["label"]])
    label = CLAIMS[key]["label"]
    pattern = {"api": rf"^- {re.escape(label)}: (.+)$", "deps": rf"^(?:# )?{re.escape(label)}(?:==|: )(.+)$"}[topic]
    m = re.search(pattern, text, re.M)
    return normalize(key, m.group(1)) if m else None


def source_kind(topic: str) -> str:
    return "test" if topic == "impl" else "config"


# --------------------------------------------------------------------------------------------
# Scenarios. Steps: agent role names, "CHANGE" (environment moves to its next version) or
# "CHECK" (the harness reads the environment and records an external verification event).
# Patterns give the view each agent step sees, in order; one is drawn per task.
# Views: current | stale | decoy | stale_log | check_output | none
# --------------------------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "d01_stale_config": {"topics": ["config"], "timeline": "changed", "steps": ["planner", "coder", "verifier"],
                         "patterns": ["stale current current", "current stale current", "stale stale current"]},
    "d02_incorrect_implementation": {"topics": ["impl"], "timeline": "static", "steps": ["planner", "coder", "verifier"],
                                     "patterns": ["decoy current current", "decoy decoy current", "current decoy current"]},
    "d03_dependency_update": {"topics": ["deps"], "timeline": "mid",
                              "steps": ["coder", "CHANGE", "planner", "coder", "verifier"],
                              "patterns": ["current stale current current", "current current current stale_log",
                                           "current stale stale current"]},
    "d04_conflicting_agents": {"topics": ["api", "config"], "timeline": "changed", "steps": ["planner", "coder", "verifier"],
                               "patterns": ["current stale current", "stale current stale", "current stale stale",
                                            "stale current current"]},
    "d05_same_agent_correction": {"topics": ["api", "impl"], "timeline": "changed",
                                  "steps": ["verifier", "CHECK", "verifier", "planner", "coder"],
                                  "patterns": ["stale_log check_output stale current", "stale_log check_output current stale"]},
    "d06_wrong_verifier": {"topics": ["config", "impl"], "timeline": "changed", "steps": ["planner", "coder", "verifier"],
                           "patterns": ["current current stale_log", "current current decoy", "current stale stale_log"]},
    "d07_lower_trust_correct": {"topics": ["api", "deps"], "timeline": "changed", "steps": ["planner", "coder", "verifier"],
                                "patterns": ["current stale stale_log", "current stale stale", "current decoy stale"]},
    "d08_expiring_claims": {"topics": ["config", "api"], "timeline": "override", "steps": ["planner", "coder", "verifier"],
                            "patterns": ["current current current", "current stale current", "current current stale"]},
    "d09_repeated_misinformation": {"topics": ["api", "impl"], "timeline": "changed",
                                    "steps": ["planner", "planner", "planner", "coder", "verifier"],
                                    "patterns": ["stale stale stale current current", "decoy decoy decoy current current",
                                                 "stale stale stale none current"]},
    "d10_evidence_backed_correction": {"topics": ["deps", "config"], "timeline": "changed",
                                       "steps": ["planner", "coder", "CHECK", "verifier"],
                                       "patterns": ["stale stale check_output", "stale current check_output",
                                                    "decoy stale check_output"]},
}


def _stamp(day: int, minute: int) -> str:
    return (EPOCH + timedelta(days=day, minutes=minute)).isoformat(timespec="minutes").replace("+00:00", "Z")


def step_minute(step: int) -> int:
    return 10 * (step + 1)


def _timeline(kind: str, old: Any, new: Any, steps: list[str]) -> list[dict[str, Any]]:
    """Versions of the fact as (start minute, value). Minute is relative to the task's day."""
    if kind == "static":
        return [{"start": -4320, "value": new}]
    if kind == "changed":
        return [{"start": -4320, "value": old}, {"start": CHANGE_MIN, "value": new}]
    if kind == "mid":
        return [{"start": -4320, "value": old}, {"start": step_minute(steps.index("CHANGE")), "value": new}]
    # override: `new` is the normal value, `old` a temporary override that ends at EXPIRY_MIN.
    return [{"start": -4320, "value": new}, {"start": CHANGE_MIN, "value": old, "override_until": EXPIRY_MIN},
            {"start": EXPIRY_MIN, "value": new}]


def version_at(task: dict[str, Any], minute: int) -> int:
    return max(i for i, v in enumerate(task["timeline"]) if v["start"] <= minute)


def truth_at(task: dict[str, Any], minute: int) -> str:
    """Ground truth: render the environment as it is at `minute` and read it back with `check`."""
    return check(task["topic"], file_at(task, version_at(task, minute)), task["claim_key"])


def file_at(task: dict[str, Any], version: int) -> str:
    v = task["timeline"][version]
    values = {**task["background"], task["claim_key"]: v["value"]}
    note = ""
    if "override_until" in v:
        until = _stamp(task["day"], v["override_until"])
        note = f"# NOTE: {CLAIMS[task['claim_key']]['label']} is a temporary override until {until}; normal value {task['timeline'][0]['value']}"
    return render(task["topic"], values, _stamp(task["day"] - 3 if v["start"] < 0 else task["day"], max(v["start"], 0)), note)


def previous_true_values(task: dict[str, Any], minute: int) -> list[str]:
    cur = version_at(task, minute)
    now = truth_at(task, minute)
    seen = {check(task["topic"], file_at(task, i), task["claim_key"]) for i in range(cur)}
    return sorted(v for v in seen if v is not None and v != now)


def view_document(task: dict[str, Any], view: str, minute: int) -> dict[str, str] | None:
    """The document an agent step is shown. Titles never say whether a document is current."""
    key, topic, day = task["claim_key"], task["topic"], task["day"]
    label, path = CLAIMS[key]["label"], PATHS[topic]
    cur = version_at(task, minute)
    if view == "none":
        return None
    if view == "current":
        return {"title": path, "text": file_at(task, cur)}
    if view == "stale":
        if cur == 0:
            raise ValueError(f"{task['task_id']}: no earlier version for a stale view")
        return {"title": f"{path} (copy in the team handbook)", "text": file_at(task, cur - 1)}
    if view == "decoy":
        return {"title": f"docs/design/{key.split('.', 1)[1]}.md",
                "text": f"# Design notes ({_stamp(day - 1, 0)})\nWe use {label} = {task['decoy']}.\n"}
    if view == "stale_log":
        old = task["timeline"][max(cur - 1, 0)]["value"]
        return {"title": "ci/test-report.txt",
                "text": f"# test run {_stamp(day - 2, 0)}\ntest_{label.lower().replace(' ', '_')} PASSED  ({label} == {old})\n"}
    if view == "mirror":
        # Adaptive-trust stress experiment: the verifier's package mirror stopped syncing, so it
        # shows the version before today's change. Its header gives its last-sync date.
        if cur == 0:
            raise ValueError(f"{task['task_id']}: no earlier version for a mirror view")
        synced = _stamp(task["mirror_frozen_day"], 0)
        values = {**task["background"], key: task["timeline"][cur - 1]["value"]}
        text = render(topic, values, synced).replace("(last modified", "(mirror copy, last synced", 1)
        return {"title": f"mirror/{path} (internal package mirror)", "text": text}
    if view == "check_output":
        value = task["timeline"][cur]["value"]
        return {"title": "ci/check-report.txt",
                "text": f"# check run {_stamp(day, minute - 10)}\ncheck {label}: read {path} -> {label} = {value}\n"}
    raise ValueError(view)


# --------------------------------------------------------------------------------------------
# Task generation (the frozen part)
# --------------------------------------------------------------------------------------------


def generate_tasks(split: str, seed: str, n: int) -> list[dict[str, Any]]:
    rng = random.Random(f"benchmark-d:{seed}")
    names = list(SCENARIOS)
    per = [n // len(names) + (1 if i < n % len(names) else 0) for i in range(len(names))]
    tasks: list[dict[str, Any]] = []
    for name, count in zip(names, per, strict=True):
        sc = SCENARIOS[name]
        for j in range(count):
            topic = sc["topics"][j % len(sc["topics"])]
            key = rng.choice(keys_for(topic))
            old, new, decoy = rng.sample(CLAIMS[key]["values"], 3)
            background = {k: rng.choice(CLAIMS[k]["values"]) for k in keys_for(topic) if k != key}
            tasks.append({
                "origin": "controlled_env",
                "split": split, "scenario": name, "topic": topic, "claim_key": key,
                "project": rng.choice(PROJECTS), "query": rng.choice(CLAIMS[key]["queries"]),
                "steps": sc["steps"], "views": rng.choice(sc["patterns"]).split(),
                "timeline": _timeline(sc["timeline"], old, new, sc["steps"]),
                "decoy": decoy, "background": background,
            })
    rng.shuffle(tasks)  # one interleaved stream; the same order is used for every model
    for day, t in enumerate(tasks):
        t["day"], t["position"] = day, day
        t["task_id"] = f"{split}-{day:03d}-{t['scenario']}"
        t["truth"] = truth_at(t, QUERY_MIN)
        t["stale_values"] = previous_true_values(t, QUERY_MIN)
        t["source_kind"] = source_kind(t["topic"])
        t["checks_before_query"] = [step_minute(i) for i, s in enumerate(t["steps"]) if s == "CHECK"]
    return tasks


# --------------------------------------------------------------------------------------------
# Secondary experiment: adaptive-trust stress (results/benchmark_d/adaptive_trust_stress/)
# --------------------------------------------------------------------------------------------

STRESS_DIR = ROOT / "results" / "benchmark_d" / "adaptive_trust_stress"
STRESS_DEGRADED, STRESS_CONTROL = "deps", "config"


def generate_stress_tasks(seed: str, phase1: dict[str, int], phase2: dict[str, int]) -> list[dict[str, Any]]:
    """One stream per model. Phase 1: every source healthy. Phase 2: the verifier's package mirror
    (dependencies only) has stopped syncing; config stays healthy as the control topic.
    In each task one lower-prior agent (planner or coder, drawn by seed) writes, then the verifier.
    Agents are independent: neither sees the other's note. Nobody is told what is true."""
    rng = random.Random(f"benchmark-d-stress:{seed}")

    def block(counts: dict[str, int], phase: int) -> list[dict[str, Any]]:
        out = []
        for topic, n in counts.items():
            for _ in range(n):
                key = rng.choice(keys_for(topic))
                old, new, decoy = rng.sample(CLAIMS[key]["values"], 3)
                low = rng.choice(["planner", "coder"])
                if phase == 1:
                    views = [rng.choice(["current", "stale"]), "current"]
                elif topic == STRESS_DEGRADED:
                    views = ["current", "mirror"]
                else:
                    views = ["stale", "current"]
                out.append({
                    "origin": "controlled_env", "split": "stress", "experiment": "adaptive_trust_stress",
                    "phase": phase, "scenario": f"p{phase}_{topic}_" + ("healthy" if phase == 1 else "mirror_frozen" if topic == STRESS_DEGRADED else "control"),
                    "topic": topic, "claim_key": key, "project": rng.choice(PROJECTS),
                    "query": rng.choice(CLAIMS[key]["queries"]), "steps": [low, "verifier"], "views": views,
                    "lower_agent": low, "independent": True,
                    "timeline": _timeline("changed", old, new, [low, "verifier"]), "decoy": decoy,
                    "background": {k: rng.choice(CLAIMS[k]["values"]) for k in keys_for(topic) if k != key},
                })
        rng.shuffle(out)
        return out

    tasks = block(phase1, 1) + block(phase2, 2)  # phase order is fixed; order within a phase is seeded
    frozen_day = sum(phase1.values()) - 1  # the mirror last synced on the final phase-1 day
    for day, t in enumerate(tasks):
        t["day"], t["position"] = day, day
        t["mirror_frozen_day"] = frozen_day
        t["task_id"] = f"stress-{day:03d}-{t['scenario']}"
        t["truth"] = truth_at(t, QUERY_MIN)
        t["stale_values"] = previous_true_values(t, QUERY_MIN)
        t["source_kind"] = source_kind(t["topic"])
        t["checks_before_query"] = []
    return tasks


def tasks_path(split: str) -> Path:
    return STRESS_DIR / "tasks.jsonl" if split == "stress" else TASKS_DIR / f"{split}.jsonl"


def write_tasks(split: str, seed: str, n: int, tasks: list[dict[str, Any]] | None = None) -> Path:
    path = tasks_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in tasks if tasks is not None else generate_tasks(split, seed, n):
            f.write(json.dumps(t, sort_keys=True) + "\n")
    return path


def load_tasks(split: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in tasks_path(split).open()]


def at(task: dict[str, Any], minute: int) -> datetime:
    return EPOCH + timedelta(days=task["day"], minutes=minute)
