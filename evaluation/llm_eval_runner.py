"""Run LLM evaluation: N tasks × memory techniques via xAI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baselines.factory import create_backend
from baselines.runtime import configure_torch_safety, warn_if_risky_benchmark_env
from baselines.techniques import list_techniques, technique_label
from evaluation.harness import _detect_git_sha
from evaluation.llm_complex_suite import EvalSuite, get_suite, task_success_score
from evaluation.llm_fixtures import LLMEvalTask, has_forbidden, term_recall
from evaluation.llm_judge import generate_answer, judge_answer, parse_judge_score
from retrieval.context import RetrievalContext
from storage import init_db, session_scope
from storage.repository import MemoryRepository
from utils.config import load_config, reload_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class LLMCellResult:
    technique: str
    task_id: str
    asker: str
    query: str
    retrieved_context: str
    retrieved_ids: list[str] = field(default_factory=list)
    retrieval_term_recall: float = 0.0
    answer: str = ""
    answer_term_recall: float = 0.0
    task_success: float = 0.0
    forbidden_in_answer: bool = False
    judge: dict[str, Any] = field(default_factory=dict)
    judge_score: float = 0.0
    backend: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class LLMEvalSummary:
    run_id: str
    model: str
    seed: int
    suite: str
    cells: list[LLMCellResult] = field(default_factory=list)
    git_sha: str | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def by_technique(self) -> dict[str, list[LLMCellResult]]:
        out: dict[str, list[LLMCellResult]] = {}
        for cell in self.cells:
            out.setdefault(cell.technique, []).append(cell)
        return out

    def mean_judge_by_technique(self) -> dict[str, float]:
        grouped = self.by_technique()
        return {
            tech: sum(c.judge_score for c in rows) / len(rows) if rows else 0.0
            for tech, rows in grouped.items()
        }

    def mean_task_success_by_technique(self) -> dict[str, float]:
        grouped = self.by_technique()
        return {
            tech: sum(c.task_success for c in rows) / len(rows) if rows else 0.0
            for tech, rows in grouped.items()
        }


def _isolated_config() -> tuple[Any, str]:
    base = load_config()
    tmp = tempfile.mkdtemp(prefix="llm_eval_")
    offline = os.getenv("MEMORY_OFFLINE_RETRIEVAL", "1").lower() in ("1", "true", "yes")
    cfg = base.model_copy(
        update={
            "database_url": "sqlite:///:memory:",
            "faiss_index_path": str(Path(tmp) / "faiss.index"),
            "offline_retrieval": offline,
        }
    )
    return cfg, tmp


def run_cell(
    technique: str,
    task: LLMEvalTask,
    suite: EvalSuite,
    *,
    mem0_cache_dir: Path | None,
    model: str | None,
    top_k: int = 6,
) -> LLMCellResult:
    from benchmarks._helpers import reset_storage_engine

    configure_torch_safety()
    cfg, tmp_dir = _isolated_config()
    reset_storage_engine()
    init_db(cfg)

    cell = LLMCellResult(
        technique=technique,
        task_id=task.id,
        asker=task.asker,
        query=task.query,
        retrieved_context="",
    )

    try:
        with session_scope(cfg) as session:
            repo = MemoryRepository(session)
            suite.seed_fn(repo)
            backend = create_backend(
                technique,
                repo,
                cfg,
                work_dir=Path(tmp_dir),
                mem0_cache_dir=mem0_cache_dir,
            )
            cell.backend = backend.backend_info()
            ctx = RetrievalContext(agent_id=task.asker, task_id=suite.task_id)
            result = backend.retrieve(task.query, ctx, top_k=top_k)
            hits = list(result.hits)
            backend.close()

        lines: list[str] = []
        for i, hit in enumerate(hits[:top_k], start=1):
            content = hit.get("content", "")
            agent = hit.get("agent", "?")
            scope = hit.get("scope", "?")
            score = hit.get("score", hit.get("final_score", 0.0))
            lines.append(f"{i}. [{scope}/{agent}] (score={score:.3f}) {content}")
        context = "\n".join(lines) if lines else "(no memories retrieved)"
        cell.retrieved_context = context
        cell.retrieved_ids = [str(h.get("memory_id", "")) for h in hits]
        cell.retrieval_term_recall = term_recall(task.required_terms, context)

        cell.answer = generate_answer(
            task.query,
            context,
            technique=technique,
            model=model,
        )
        cell.answer_term_recall = term_recall(task.required_terms, cell.answer)
        cell.task_success = task_success_score(cell.answer, task)
        cell.forbidden_in_answer = has_forbidden(task.forbidden_terms, cell.answer)

        cell.judge = judge_answer(
            task_id=task.id,
            query=task.query,
            required_terms=task.required_terms,
            forbidden_terms=task.forbidden_terms,
            memory_context=context,
            answer=cell.answer,
            model=model,
        )
        cell.judge_score = parse_judge_score(cell.judge)

    except Exception as exc:
        cell.error = str(exc)
        logger.exception("LLM eval cell failed: %s × %s", technique, task.id)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        reset_storage_engine()

    return cell


def _prewarm_retrieval_models(cfg: Any) -> None:
    """Load BGE + reranker once per sweep so cells do not re-hit Hugging Face."""
    if cfg.offline_retrieval:
        return
    from retrieval.embedder import MemoryEmbedder
    from retrieval.reranker import CrossEncoderReranker

    logger.info("Pre-warming embedding model: %s", cfg.embedding_model)
    MemoryEmbedder(cfg.embedding_model, offline=False).encode_query("warmup")
    if cfg.use_reranker:
        logger.info("Pre-warming reranker: %s", cfg.reranker_model)
        CrossEncoderReranker(cfg.reranker_model, enabled=True, offline=False).predict_scores(
            "warmup",
            [{"content": "warmup memory"}],
        )


def run_evaluation(
    *,
    techniques: list[str] | None = None,
    model: str | None = None,
    seed: int = 42,
    suite_name: str = "complex",
    runs_dir: Path | str | None = None,
) -> LLMEvalSummary:
    from evaluation.llm_judge import default_model

    reload_config()
    cfg = load_config()
    for line in warn_if_risky_benchmark_env():
        logger.warning(line)
    _prewarm_retrieval_models(cfg)

    suite = get_suite(suite_name)
    techs = techniques or list_techniques()
    run_id = f"llm_eval_{suite.name}_seed{seed}"
    out_root = Path(runs_dir or cfg.experiments_runs_dir) / "llm_eval" / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    mem0_cache = out_root / "mem0_cache"

    summary = LLMEvalSummary(
        run_id=run_id,
        model=model or default_model(),
        seed=seed,
        suite=suite.name,
        git_sha=_detect_git_sha(),
    )

    total = len(techs) * len(suite.tasks)
    done = 0
    for technique in techs:
        for task in suite.tasks:
            done += 1
            logger.info(
                "LLM eval [%d/%d]: %s × %s",
                done,
                total,
                technique,
                task.id,
            )
            cell = run_cell(
                technique,
                task,
                suite,
                mem0_cache_dir=mem0_cache if technique == "mem0" else None,
                model=model,
            )
            summary.cells.append(cell)

    _write_artifacts(summary, suite, out_root)
    _print_report(summary)
    return summary


def _write_artifacts(summary: LLMEvalSummary, suite: EvalSuite, out_dir: Path) -> None:
    payload = {
        "run_id": summary.run_id,
        "model": summary.model,
        "seed": summary.seed,
        "suite": summary.suite,
        "git_sha": summary.git_sha,
        "started_at": summary.started_at,
        "tasks": [asdict(t) for t in suite.tasks],
        "cells": [asdict(c) for c in summary.cells],
        "mean_judge_by_technique": summary.mean_judge_by_technique(),
        "mean_task_success_by_technique": summary.mean_task_success_by_technique(),
    }
    json_path = out_dir / f"{summary.run_id}_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    md_lines = [
        f"# LLM evaluation ({summary.run_id})",
        "",
        f"- Model: `{summary.model}`",
        f"- Suite: `{summary.suite}` ({len(suite.tasks)} tasks per technique)",
        f"- Techniques: {', '.join(summary.mean_judge_by_technique().keys())}",
        "",
        "## Mean judge score by technique",
        "",
        "| Technique | Mean judge | Mean task success |",
        "|-----------|------------|-------------------|",
    ]
    judge_means = summary.mean_judge_by_technique()
    success_means = summary.mean_task_success_by_technique()
    for tech in sorted(judge_means.keys(), key=lambda t: (-judge_means[t], t)):
        md_lines.append(
            f"| {technique_label(tech)} | {judge_means[tech]:.3f} | "
            f"{success_means.get(tech, 0.0):.3f} |"
        )
    md_lines.append("")
    md_lines.append("## Per-cell results")
    md_lines.append("")
    for cell in summary.cells:
        md_lines.append(f"### {technique_label(cell.technique)} — `{cell.task_id}`")
        if cell.error:
            md_lines.append(f"\n**Error:** {cell.error}\n")
            continue
        md_lines.append(f"\n**Judge score:** {cell.judge_score:.3f}")
        md_lines.append(f"\n**Task success:** {cell.task_success:.3f}")
        md_lines.append(f"\n**Retrieval term recall:** {cell.retrieval_term_recall:.3f}")
        md_lines.append(f"\n**Answer term recall:** {cell.answer_term_recall:.3f}")
        md_lines.append(f"\n**Question:** {cell.query}")
        md_lines.append(f"\n**Answer:**\n\n{cell.answer}")
        md_lines.append(f"\n**Judge reasoning:** {cell.judge.get('reasoning', '')}")
        md_lines.append("")
    (out_dir / f"{summary.run_id}_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def _print_report(summary: LLMEvalSummary) -> None:
    print(f"\n{'=' * 72}")
    print(
        f"LLM evaluation complete | suite={summary.suite} | model={summary.model} | "
        f"run={summary.run_id}"
    )
    print(f"{'=' * 72}\n")
    print("| Technique | Task | Judge | Success | Retrieval |")
    print("|-----------|------|-------|---------|-----------|")
    for cell in summary.cells:
        if cell.error:
            print(
                f"| {technique_label(cell.technique):<20} | {cell.task_id:<22} | ERROR | - | - |"
            )
            continue
        print(
            f"| {technique_label(cell.technique):<20} | {cell.task_id:<22} | "
            f"{cell.judge_score:.2f} | {cell.task_success:.2f} | "
            f"{cell.retrieval_term_recall:.2f} |"
        )
    print("\nMean judge score by technique:")
    for tech, score in sorted(
        summary.mean_judge_by_technique().items(),
        key=lambda x: (-x[1], x[0]),
    ):
        succ = summary.mean_task_success_by_technique().get(tech, 0.0)
        print(f"  {technique_label(tech):<28} judge={score:.3f}  success={succ:.3f}")
    print()


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(
        description="LLM eval: memory techniques via xAI Grok (basic=2 tasks, complex=10)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="xAI model (default: XAI_MODEL)")
    parser.add_argument(
        "--suite",
        choices=("basic", "complex"),
        default="complex",
        help="Task suite (default: complex, 10 hard tasks)",
    )
    parser.add_argument("--technique", action="append")
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args(argv)

    reload_config()
    if not os.getenv("XAI_API_KEY", "").strip():
        raise SystemExit("Set XAI_API_KEY in .env.local before running LLM evaluation.")

    run_evaluation(
        techniques=args.technique,
        model=args.model,
        seed=args.seed,
        suite_name=args.suite,
        runs_dir=args.runs_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
