"""Benchmark B: harder scenarios for the real-run experiment.

Kept separate from Benchmark A (``real_run.py``, 40 clean-correction tasks).
Tasks live in ``experiments/hard_tasks.yaml``; each one fixes its correct fact
(``expected``) before any model runs. Four categories:

  multi_version   several successive updates; the newest is true
  ambiguous       the update never says it replaces the old note
  conflicting     two agents disagree and nobody resolves it
  wrong_verifier  the verifier is confident and wrong; the planner is right

Every note is labelled with the fact it was generated from, so a hit is scored as
``correct`` (key note of the expected fact), ``wrong`` (key note of another fact)
or ``miss`` (an unrelated note). Results are reported per category and model,
never pooled with Benchmark A.

Usage:
  python -m experiments.hard_run validate                      # check the task file
  python -m experiments.hard_run all --provider xai --provider openai
  python -m experiments.hard_run evaluate --provider xai        # re-score from cache
  python -m experiments.hard_run all --provider fake --offline  # no API at all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from experiments.real_run import CONDITIONS, LLM, PROVIDERS, ROLES, _ablation, wilson

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "experiments" / "hard_tasks.yaml"
RUNS_DIR = ROOT / "experiments" / "runs" / "hard_run"
CATEGORIES = ("multi_version", "ambiguous", "conflicting", "wrong_verifier")

PROMPTS = {
    "belief": "Project: {context}\nYour current understanding: {fact}",
    "build": (
        "Project: {context}\nThe latest note from the {prev_role} says: \"{prev_note}\"\n"
        "Record what you are building based on it."
    ),
    "correct": (
        "Project: {context}\nYou verified the facts with the source of truth. "
        "Earlier notes assumed: {prev_fact}\nWhat you confirmed: {fact}"
    ),
    "implicit": "Project: {context}\nRecord this for the team: {fact}",
    "decide": "Project: {context}\nAfter the earlier notes, the team made a new decision: {fact}",
    "dispute": (
        "Project: {context}\nAnother agent noted: \"{prev_note}\"\n"
        "You checked yourself and believe otherwise: {fact}"
    ),
}


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text())["tasks"]


def validate(tasks: list[dict[str, Any]]) -> list[str]:
    """Return every problem with the task file; empty means it is usable."""
    errors: list[str] = []
    ids = [t.get("id") for t in tasks]
    errors += [f"duplicate id {i}" for i in set(ids) if ids.count(i) > 1]
    for t in tasks:
        tid = t.get("id", "?")
        for field in ("category", "context", "facts", "writes", "expected", "why", "query"):
            if field not in t:
                errors.append(f"{tid}: missing {field}")
        if t.get("category") not in CATEGORIES:
            errors.append(f"{tid}: unknown category {t.get('category')!r}")
        facts, writes = t.get("facts", {}), t.get("writes", [])
        if t.get("expected") not in facts:
            errors.append(f"{tid}: expected {t.get('expected')!r} is not a fact")
        if len(facts) < 2:
            errors.append(f"{tid}: needs at least two competing facts")
        written = {w.get("fact") for w in writes}
        errors += [f"{tid}: fact {f} is never written" for f in facts if f not in written]
        for i, w in enumerate(writes):
            if w.get("role") not in ROLES:
                errors.append(f"{tid}: write {i} has unknown role {w.get('role')!r}")
            if w.get("fact") not in facts:
                errors.append(f"{tid}: write {i} uses unknown fact {w.get('fact')!r}")
            if w.get("kind") not in PROMPTS:
                errors.append(f"{tid}: write {i} has unknown kind {w.get('kind')!r}")
            if i == 0 and w.get("kind") in ("build", "correct", "dispute"):
                errors.append(f"{tid}: write 0 cannot refer to a previous write")
    return errors


def prompt_for(task: dict[str, Any], i: int, notes: list[dict[str, str]]) -> str:
    w = task["writes"][i]
    prev = task["writes"][i - 1] if i else None
    return PROMPTS[w["kind"]].format(
        context=task["context"],
        fact=task["facts"][w["fact"]],
        prev_role=prev["role"] if prev else "",
        prev_fact=task["facts"][prev["fact"]] if prev else "",
        prev_note=notes[i - 1]["key_note"] if prev else "",
    )


def generate(provider: str, run_dir: Path, tasks: list[dict[str, Any]], retries: int = 3) -> Path:
    llm = LLM(provider)
    out = run_dir / "generations.jsonl"
    done = {json.loads(line)["task_id"] for line in out.open()} if out.exists() else set()
    with out.open("a") as fh:
        for task in tasks:
            if task["id"] in done:
                continue
            notes: list[dict[str, str]] = []
            for i, w in enumerate(task["writes"]):
                if provider == "fake":  # echo the labelled fact; tests the scoring, not the model
                    notes.append({"key_note": task["facts"][w["fact"]], "other_note": f"{w['role']}: keep docs current."})
                    continue
                prompt = prompt_for(task, i, notes)
                for attempt in range(retries):
                    try:
                        notes.append(llm.notes(w["role"], prompt))
                        break
                    except (ValueError, json.JSONDecodeError) as exc:
                        if attempt == retries - 1:
                            raise SystemExit(f"{task['id']}/write {i}: bad reply after {retries} tries: {exc}")
            row = {"task_id": task["id"], "provider": provider, "model": llm.model, "notes": notes}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{provider}] generated {task['id']}")
    return out


def run_condition(condition: str, rows: list[dict], tasks: dict[str, dict]) -> list[dict]:
    """Insert every task's notes into a fresh store and query each task once."""
    from baselines.factory import create_backend
    from baselines.runtime import configure_torch_safety
    from benchmarks._helpers import at, make_bench_config, reset_storage_engine
    from memory.schema import Memory, MemoryScope
    from retrieval.context import RetrievalContext
    from storage import init_db, session_scope
    from storage.repository import MemoryRepository
    from storage.sqlite_store import SQLiteMemoryStore

    technique, ablation_name = CONDITIONS[condition]
    configure_torch_safety()
    reset_storage_engine()
    tmp = Path(tempfile.mkdtemp(prefix=f"hard_run_{condition}_"))
    results: list[dict] = []
    try:
        config = make_bench_config(f"hard_run_{condition}", faiss_dir=tmp)
        init_db(config)
        with session_scope(config) as session:
            store = SQLiteMemoryStore(session)
            backend = create_backend(
                technique, MemoryRepository(session), config, work_dir=tmp, ablation=_ablation(ablation_name)
            )
            labels: dict[str, dict[str, tuple[str, str | None]]] = {}
            for row in rows:
                task = tasks[row["task_id"]]
                for i, (w, note) in enumerate(zip(task["writes"], row["notes"])):
                    for j, field in enumerate(("key_note", "other_note")):
                        mem = store.create(
                            Memory(
                                content=note[field],
                                agent=w["role"],
                                scope=MemoryScope.SHARED,
                                task_id=task["id"],
                                confidence=ROLES[w["role"]],
                                importance=0.8 if field == "key_note" else 0.5,
                                timestamp=at(minutes=10 * i + j),
                            )
                        )
                        fact = w["fact"] if field == "key_note" else None
                        labels.setdefault(task["id"], {})[str(mem.id)] = (w["role"], fact)
                backend.sync_from_repository(task_id=task["id"])

            for row in rows:
                task = tasks[row["task_id"]]
                lab = labels[task["id"]]
                result = backend.retrieve(task["query"], RetrievalContext(agent_id="reader", task_id=task["id"]), top_k=5)
                ids = [h["memory_id"] for h in result.hits if h["memory_id"] in lab]
                top_fact = lab[ids[0]][1] if ids else None
                outcome = "miss" if top_fact is None else ("correct" if top_fact == task["expected"] else "wrong")
                results.append(
                    {
                        "task_id": task["id"],
                        "category": task["category"],
                        "condition": condition,
                        "outcome": outcome,
                        "top_role": lab[ids[0]][0] if ids else None,
                        "top_fact": top_fact,
                        "ranked": [{"role": lab[i][0], "fact": lab[i][1]} for i in ids],
                    }
                )
            backend.close()
    finally:
        reset_storage_engine()
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def summarize(results: list[dict]) -> dict[str, dict[str, dict[str, Any]]]:
    """category -> condition -> rates. Never pooled across categories or benchmarks."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cat in CATEGORIES:
        for cond in CONDITIONS:
            rows = [r for r in results if r["category"] == cat and r["condition"] == cond]
            if not rows:
                continue
            n = len(rows)
            counts = {k: sum(r["outcome"] == k for r in rows) for k in ("correct", "wrong", "miss")}
            out.setdefault(cat, {})[cond] = {
                "n": n,
                **{f"{k}_at_1": v / n for k, v in counts.items()},
                "wrong_at_1_ci95": wilson(counts["wrong"], n),
            }
    return out


def print_summary(label: str, summary: dict) -> None:
    print(f"\n== {label} ==")
    print(f"{'category':<16}{'condition':<14}{'n':>4}{'correct':>9}{'wrong':>8}{'miss':>7}")
    for cat, conds in summary.items():
        for cond, s in conds.items():
            print(f"{cat:<16}{cond:<14}{s['n']:>4}{s['correct_at_1']:>9.2f}{s['wrong_at_1']:>8.2f}{s['miss_at_1']:>7.2f}")


def evaluate(run_dir: Path, tasks: list[dict[str, Any]]) -> dict:
    gen = run_dir / "generations.jsonl"
    if not gen.exists():
        raise SystemExit(f"No generations at {gen}; run `generate` first.")
    by_id = {t["id"]: t for t in tasks}
    rows = [r for r in (json.loads(line) for line in gen.open()) if r["task_id"] in by_id]
    results = [r for cond in CONDITIONS for r in run_condition(cond, rows, by_id)]
    summary = summarize(results)
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(run_dir.name, summary)
    return summary


def label_sheet(run_dir: Path, tasks: list[dict[str, Any]]) -> Path:
    """CSV of every generated key note next to the fact it should state, for checking by hand.

    A model can paraphrase a fact into something else; this sheet is how you catch it.
    """
    by_id = {t["id"]: t for t in tasks}
    out = run_dir / "label_check.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task_id", "write", "role", "fact_id", "fact", "key_note", "states_fact_yes_no"])
        for row in (json.loads(line) for line in (run_dir / "generations.jsonl").open()):
            task = by_id.get(row["task_id"])
            if not task:
                continue
            for i, (w, note) in enumerate(zip(task["writes"], row["notes"])):
                writer.writerow([task["id"], i, w["role"], w["fact"], task["facts"][w["fact"]], note["key_note"], ""])
    print(f"label check sheet: {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["validate", "generate", "evaluate", "all"])
    parser.add_argument("--provider", action="append", choices=[*PROVIDERS, "fake"], default=[])
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    parser.add_argument("--offline", action="store_true", help="hash embeddings, no model downloads (smoke test)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    errors = validate(tasks)
    if errors:
        print("\n".join(errors))
        return 1
    if args.command == "validate":
        print(f"{len(tasks)} tasks OK: " + ", ".join(f"{c} {sum(t['category'] == c for t in tasks)}" for c in CATEGORIES))
        return 0
    if not args.provider:
        parser.error("--provider is required")

    from utils.config import load_config

    load_config()
    os.environ["MEMORY_OFFLINE_RETRIEVAL"] = "1" if args.offline else "0"
    tasks = tasks[: args.limit]
    for provider in args.provider:
        run_dir = args.runs_dir / provider
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.command in ("generate", "all"):
            generate(provider, run_dir, tasks)
            label_sheet(run_dir, tasks)
        if args.command in ("evaluate", "all"):
            evaluate(run_dir, tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
