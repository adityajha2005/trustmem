"""Real-run experiment: LLM agents write their own memories, then we count stale wins.

Unlike ``benchmarks/``, no memory text is scripted. For every task in
``experiments/real_run_tasks.yaml`` three LLM agents write to one shared store:

  planner  (confidence 0.60) - records the original fact
  coder    (confidence 0.70) - records what it is building from the plan
  verifier (confidence 0.95) - records the fact it confirmed (the correction)

Each agent returns one *key note* about the fact plus one unrelated note, so every
record is labelled by construction (no LLM labelling). There is NO contradiction
resolution step: the ranking alone must keep the correction on top.

Two phases, so the API is only paid once:
  generate  - call the LLM, cache notes to ``generations.jsonl`` (resumable)
  evaluate  - offline: load notes, run every retrieval condition, write metrics

Usage:
  python -m experiments.real_run generate --provider xai
  python -m experiments.real_run evaluate --provider xai
  python -m experiments.real_run all --provider xai --provider openai
  python -m experiments.real_run all --provider fake --offline   # no API, smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "experiments" / "real_run_tasks.yaml"
RUNS_DIR = ROOT / "experiments" / "runs" / "real_run"

ROLES = {"planner": 0.60, "coder": 0.70, "verifier": 0.95}

# condition -> (technique, ablation factory name or None)
CONDITIONS: dict[str, tuple[str, str | None]] = {
    "dense": ("simple_rag", None),  # plain vector search
    "dense_rerank": ("collaborative", "with_naive_rerank"),  # + MS MARCO cross-encoder
    "no_trust": ("collaborative", "no_trust_scoring"),  # similarity + recency, no trust
    "full": ("collaborative", "full_rerank"),  # full pipeline with trust fusion
}

# ---------------------------------------------------------------------------
# LLM providers (OpenAI-compatible chat API)
# ---------------------------------------------------------------------------

PROVIDERS = {
    # name: (key env, base url env, default base url, model env, default model)
    "xai": ("XAI_API_KEY", "XAI_BASE_URL", "https://api.x.ai/v1", "XAI_MODEL", "grok-3-mini"),
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_MODEL", None),
}

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are the {role} agent on a software team. You write notes to a shared team
    memory that other agents will search later. Write like a busy engineer: short,
    specific, in your own words. Respond with JSON only:
    {{"key_note": "<one note about the fact you were given>",
      "other_note": "<one note about some other, unrelated aspect of the project>"}}"""
)

ROLE_PROMPTS = {
    "planner": "Project: {context}\nYou are planning the work. Your current understanding: {original}",
    "coder": (
        "Project: {context}\nYou are implementing the work. The planner's note says: "
        "\"{planner_note}\"\nRecord what you are building based on it."
    ),
    "verifier": (
        "Project: {context}\nYou verified the facts with the source of truth. "
        "Earlier notes assumed: {original}\nWhat you confirmed: {correction}"
    ),
}


class LLM:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        if provider == "fake":
            self.model = "fake"
            return
        key_env, url_env, url_default, model_env, model_default = PROVIDERS[provider]
        api_key = os.getenv(key_env, "").strip()
        if not api_key:
            raise SystemExit(f"{key_env} is not set (add it to .env.local).")
        self.model = os.getenv(model_env, model_default or "").strip()
        if not self.model:
            raise SystemExit(f"{model_env} is not set (add the model name to .env.local).")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=os.getenv(url_env, url_default))

    def notes(self, role: str, user: str) -> dict[str, str]:
        if self.provider == "fake":
            return _fake_notes(role, user)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(role=role)},
                {"role": "user", "content": user},
            ],
        )
        return parse_notes(resp.choices[0].message.content or "")

    def answer(self, query: str, memories: list[str]) -> str:
        if self.provider == "fake":
            return memories[0] if memories else "I don't know."
        context = "\n".join(f"- {m}" for m in memories)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer in one sentence using ONLY these team memories."},
                {"role": "user", "content": f"Memories:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def parse_notes(text: str) -> dict[str, str]:
    """Extract the JSON object from a model reply (tolerates code fences / prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in reply: {text[:200]!r}")
    data = json.loads(text[start : end + 1])
    key, other = str(data.get("key_note", "")).strip(), str(data.get("other_note", "")).strip()
    if not key or not other:
        raise ValueError(f"Missing key_note/other_note: {data!r}")
    return {"key_note": key, "other_note": other}


def _fake_notes(role: str, user: str) -> dict[str, str]:
    """Deterministic stand-in: echoes the fact the role was given."""
    fact = re.findall(r'(?:understanding|says|confirmed): "?(.+?)"?$', user, re.M)[-1]
    if role == "verifier":
        fact = f"Confirmed: {fact}"
    return {"key_note": fact, "other_note": f"{role} note: keep the README up to date."}


# ---------------------------------------------------------------------------
# Phase 1: generate
# ---------------------------------------------------------------------------


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, str]]:
    return yaml.safe_load(path.read_text())["tasks"]


def generate(provider: str, run_dir: Path, tasks: list[dict[str, str]], retries: int = 3) -> Path:
    llm = LLM(provider)
    out = run_dir / "generations.jsonl"
    done = {json.loads(line)["task_id"] for line in out.open()} if out.exists() else set()
    with out.open("a") as fh:
        for task in tasks:
            if task["id"] in done:
                continue
            row: dict[str, Any] = {"task_id": task["id"], "provider": provider, "model": llm.model}
            for role in ROLES:
                prompt = ROLE_PROMPTS[role].format(
                    planner_note=row.get("planner", {}).get("key_note", ""), **task
                )
                for attempt in range(retries):
                    try:
                        row[role] = llm.notes(role, prompt)
                        break
                    except (ValueError, json.JSONDecodeError) as exc:
                        if attempt == retries - 1:
                            raise SystemExit(f"{task['id']}/{role}: bad reply after {retries} tries: {exc}")
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{provider}] generated {task['id']}")
    return out


# ---------------------------------------------------------------------------
# Phase 2: evaluate (offline, no API)
# ---------------------------------------------------------------------------


@dataclass
class Record:
    memory_id: str
    role: str
    is_key: bool
    content: str


def _ablation(name: str | None):
    from retrieval.ablation import RetrievalAblation

    if name is None:
        return None
    if name == "full_rerank":
        return RetrievalAblation.full(use_reranker=True)
    return getattr(RetrievalAblation, name)()


def run_condition(
    condition: str,
    rows: list[dict[str, Any]],
    tasks: dict[str, dict],
    confidence: Callable[[str, str], float] | None = None,
) -> list[dict]:
    """Insert every generation into a fresh store and query each task once.

    ``confidence(task_id, role)`` overrides the role's stored confidence (noisy-trust
    runs). Labels and metrics always use the *true* roles: the verifier is right.
    """
    from baselines.factory import create_backend
    from baselines.runtime import configure_torch_safety
    from benchmarks._helpers import at, make_bench_config, reset_storage_engine
    from evaluation.metrics import ranking_inversion_rate
    from memory.schema import Memory, MemoryScope
    from retrieval.context import RetrievalContext
    from storage import init_db, session_scope
    from storage.repository import MemoryRepository
    from storage.sqlite_store import SQLiteMemoryStore

    technique, ablation_name = CONDITIONS[condition]
    configure_torch_safety()
    reset_storage_engine()
    tmp = Path(tempfile.mkdtemp(prefix=f"real_run_{condition}_"))
    results: list[dict] = []
    try:
        config = make_bench_config(f"real_run_{condition}", faiss_dir=tmp)
        init_db(config)
        with session_scope(config) as session:
            store = SQLiteMemoryStore(session)
            backend = create_backend(
                technique,
                MemoryRepository(session),
                config,
                work_dir=tmp,
                ablation=_ablation(ablation_name),
            )
            records: dict[str, list[Record]] = {}
            for row in rows:
                tid = row["task_id"]
                # Writes happen in role order, a few minutes apart (verifier is last).
                for i, (role, conf) in enumerate(ROLES.items()):
                    for j, field in enumerate(("key_note", "other_note")):
                        mem = store.create(
                            Memory(
                                content=row[role][field],
                                agent=role,
                                scope=MemoryScope.SHARED,
                                task_id=tid,
                                confidence=confidence(tid, role) if confidence else conf,
                                importance=0.8 if field == "key_note" else 0.5,
                                timestamp=at(minutes=10 * i + j),
                            )
                        )
                        records.setdefault(tid, []).append(
                            Record(str(mem.id), role, field == "key_note", mem.content)
                        )
                backend.sync_from_repository(task_id=tid)

            for row in rows:
                tid = row["task_id"]
                recs = {r.memory_id: r for r in records[tid]}
                ctx = RetrievalContext(agent_id="reader", task_id=tid)
                result = backend.retrieve(tasks[tid]["query"], ctx, top_k=5)
                ids = [h["memory_id"] for h in result.hits]
                key_ids = [i for i in ids if i in recs and recs[i].is_key]
                conf = {i: ROLES[recs[i].role] for i in key_ids}
                top = recs.get(ids[0]) if ids else None
                results.append(
                    {
                        "task_id": tid,
                        "condition": condition,
                        "correction_at_1": bool(top and top.is_key and top.role == "verifier"),
                        "stale_at_1": bool(top and top.is_key and top.role != "verifier"),
                        "inversion_rate": ranking_inversion_rate(key_ids, conf),
                        "top_role": top.role if top else None,
                        "top_content": top.content if top else None,
                        "ranked": [
                            {"role": recs[i].role, "key": recs[i].is_key, "content": recs[i].content}
                            for i in ids
                            if i in recs
                        ],
                    }
                )
            backend.close()
    finally:
        reset_storage_engine()
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(results: list[dict]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for cond in CONDITIONS:
        rows = [r for r in results if r["condition"] == cond]
        n = len(rows)
        if not n:
            continue
        stale = sum(r["stale_at_1"] for r in rows)
        correct = sum(r["correction_at_1"] for r in rows)
        summary[cond] = {
            "n": n,
            "stale_at_1": stale / n,
            "stale_at_1_ci95": wilson(stale, n),
            "correction_at_1": correct / n,
            "correction_at_1_ci95": wilson(correct, n),
            "mean_inversion_rate": sum(r["inversion_rate"] for r in rows) / n,
        }
    return summary


def evaluate(run_dir: Path, tasks: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    gen = run_dir / "generations.jsonl"
    if not gen.exists():
        raise SystemExit(f"No generations at {gen}; run `generate` first.")
    rows = [json.loads(line) for line in gen.open()]
    by_id = {t["id"]: t for t in tasks}
    rows = [r for r in rows if r["task_id"] in by_id]
    results: list[dict] = []
    for cond in CONDITIONS:
        print(f"evaluating {cond} on {len(rows)} tasks")
        results += run_condition(cond, rows, by_id)
    summary = summarize(results)
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(run_dir.name, summary)
    return summary


def print_summary(label: str, summary: dict[str, dict[str, Any]]) -> None:
    print(f"\n== {label} ==")
    print(f"{'condition':<14}{'n':>4}{'stale@1':>10}{'95% CI':>16}{'correct@1':>11}{'inversion':>11}")
    for cond, s in summary.items():
        lo, hi = s["stale_at_1_ci95"]
        print(
            f"{cond:<14}{s['n']:>4}{s['stale_at_1']:>10.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
            f"{s['correction_at_1']:>11.2f}{s['mean_inversion_rate']:>11.2f}"
        )


# ---------------------------------------------------------------------------
# Noisy trust (offline): what if the confidence signal is wrong?
# ---------------------------------------------------------------------------


def noise_settings(seed: int = 42) -> dict[str, Callable[[str, str], float]]:
    """Confidence overrides. ``flip_p``: on a random p of tasks the coder (stale) gets the
    verifier's 0.95 and the verifier gets 0.70 - a confidently wrong agent. ``equal``: no
    trust signal at all. ``jitter``: every confidence +/- Gaussian noise (sd 0.15)."""

    def flip(p: float) -> Callable[[str, str], float]:
        def fn(tid: str, role: str) -> float:
            flipped = random.Random(f"{seed}:{tid}").random() < p
            if flipped and role in ("coder", "verifier"):
                return ROLES["verifier" if role == "coder" else "coder"]
            return ROLES[role]

        return fn

    def jitter(tid: str, role: str) -> float:
        noise = random.Random(f"{seed}:{tid}:{role}").gauss(0, 0.15)
        return min(1.0, max(0.0, ROLES[role] + noise))

    return {
        "flip_10": flip(0.10),
        "flip_25": flip(0.25),
        "flip_50": flip(0.50),
        "equal": lambda tid, role: 0.70,
        "jitter": jitter,
    }


def noisy(run_dir: Path, tasks: list[dict[str, str]], conditions=("dense_rerank", "full")) -> dict:
    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").open()]
    by_id = {t["id"]: t for t in tasks}
    rows = [r for r in rows if r["task_id"] in by_id]
    out: dict[str, dict] = {}
    for name, fn in noise_settings().items():
        results: list[dict] = []
        for cond in conditions:
            print(f"noisy {name}: {cond}")
            results += run_condition(cond, rows, by_id, confidence=fn)
        out[name] = summarize(results)
        print_summary(f"{run_dir.name} / {name}", out[name])
    (run_dir / "noisy_summary.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Phase 3 (optional): answers for hand-checking
# ---------------------------------------------------------------------------


def handcheck(provider: str, run_dir: Path, tasks: list[dict[str, str]], n: int = 20, seed: int = 42) -> Path:
    """Answer a random sample of tasks from `dense` and `full` context; CSV for hand labels."""
    llm = LLM(provider)
    by_id = {t["id"]: t for t in tasks}
    results = [json.loads(line) for line in (run_dir / "results.jsonl").open()]
    sample = set(random.Random(seed).sample(sorted({r["task_id"] for r in results}), k=n))
    out = run_dir / "handcheck.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task_id", "condition", "query", "correct_fact", "answer", "label_correct_yes_no"])
        for r in results:
            if r["task_id"] in sample and r["condition"] in ("dense", "full"):
                memories = [x["content"] for x in r["ranked"][:3]]
                t = by_id[r["task_id"]]
                writer.writerow(
                    [r["task_id"], r["condition"], t["query"], t["correction"], llm.answer(t["query"], memories), ""]
                )
    print(f"hand-check sheet: {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "evaluate", "handcheck", "noisy", "all"])
    parser.add_argument("--provider", action="append", choices=[*PROVIDERS, "fake"], required=True)
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    parser.add_argument("--handcheck-n", type=int, default=20)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="hash embeddings + no cross-encoder download (smoke test only; not real results)",
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)

    from utils.config import load_config

    load_config()  # loads .env / .env.local into the environment
    os.environ["MEMORY_OFFLINE_RETRIEVAL"] = "1" if args.offline else "0"

    tasks = load_tasks(args.tasks)[: args.limit]
    for provider in args.provider:
        run_dir = args.runs_dir / provider
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.command in ("generate", "all"):
            generate(provider, run_dir, tasks)
        if args.command in ("evaluate", "all"):
            evaluate(run_dir, tasks)
        if args.command == "noisy":
            noisy(run_dir, tasks)
        if args.command in ("handcheck", "all"):
            handcheck(provider, run_dir, tasks, n=min(args.handcheck_n, len(tasks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
