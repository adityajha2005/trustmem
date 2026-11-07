"""End-to-end demo: real memory backends + xAI Grok answers."""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from baselines.factory import create_backend
from baselines.techniques import list_techniques, technique_label
from memory.schema import Memory, MemoryScope
from retrieval.context import RetrievalContext
from storage import init_db, session_scope
from storage.repository import MemoryRepository
from utils.config import load_config, reload_config
from utils.logging import setup_logging

DEFAULT_TASK = "e2e_demo"
DEFAULT_QUERY = "What infrastructure did the team configure for deployment?"


def _seed_demo_memories(repo: MemoryRepository) -> None:
    facts = [
        (
            "agent_a",
            MemoryScope.SHARED,
            "Database connection pool configured with 50 max connections for production.",
            0.92,
        ),
        (
            "agent_b",
            MemoryScope.SHARED,
            "Cache layer uses Redis with one hour TTL for session data.",
            0.88,
        ),
        (
            "agent_c",
            MemoryScope.SHARED,
            "Message queue uses RabbitMQ with priority routing for background tasks.",
            0.9,
        ),
        (
            "agent_a",
            MemoryScope.LOCAL,
            "Personal note: agent_a debugger profile uses verbose logs.",
            1.0,
        ),
        (
            "agent_b",
            MemoryScope.LOCAL,
            "Personal note: agent_b prefers shortcut Ctrl-K for navigation.",
            1.0,
        ),
    ]
    for agent, scope, content, confidence in facts:
        repo.add(
            Memory(
                content=content,
                agent=agent,
                scope=scope,
                task_id=DEFAULT_TASK,
                confidence=confidence,
                importance=0.75,
            )
        )


def _isolated_config() -> Any:
    base = load_config()
    tmp = tempfile.mkdtemp(prefix="e2e_demo_")
    offline = os.getenv("MEMORY_OFFLINE_RETRIEVAL", "1").lower() in ("1", "true", "yes")
    return base.model_copy(
        update={
            "database_url": "sqlite:///:memory:",
            "faiss_index_path": str(Path(tmp) / "faiss.index"),
            "offline_retrieval": offline,
        }
    ), tmp


def retrieve_for_technique(
    technique: str,
    query: str,
    *,
    asker: str = "agent_b",
    top_k: int = 3,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    from benchmarks._helpers import reset_storage_engine

    cfg, tmp_dir = _isolated_config()
    reset_storage_engine()
    init_db(cfg)

    backend_info: dict[str, Any] = {}
    try:
        with session_scope(cfg) as session:
            repo = MemoryRepository(session)
            _seed_demo_memories(repo)
            backend = create_backend(
                technique,
                repo,
                cfg,
                work_dir=Path(tmp_dir),
            )
            backend_info = backend.backend_info()
            ctx = RetrievalContext(agent_id=asker, task_id=DEFAULT_TASK)
            result = backend.retrieve(query, ctx, top_k=top_k)
            hits = list(result.hits)
            backend.close()
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        reset_storage_engine()

    lines: list[str] = []
    if not hits:
        lines.append("(no memories retrieved)")
    for i, hit in enumerate(hits[:top_k], start=1):
        content = hit.get("content", "")
        agent = hit.get("agent", "?")
        scope = hit.get("scope", "?")
        score = hit.get("score", hit.get("final_score", 0.0))
        lines.append(f"{i}. [{scope}/{agent}] (score={score:.3f}) {content}")

    return "\n".join(lines), hits, backend_info


def _xai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install openai for xAI demo: pip install -e '.[llm]'") from exc

    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Set XAI_API_KEY in .env.local. Get a key at https://console.x.ai/"
        )
    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").strip()
    return OpenAI(api_key=api_key, base_url=base_url)


def answer_with_xai(
    query: str,
    memory_context: str,
    *,
    technique: str,
    model: str | None = None,
) -> str:
    client = _xai_client()
    model_name = model or os.getenv("XAI_MODEL", "grok-3-mini")

    system = textwrap.dedent(
        f"""\
        You are a helpful assistant answering from retrieved team memories only.
        Memory backend: {technique_label(technique)} (real implementation).
        If the context does not contain the answer, say so briefly.
        """
    )
    user = textwrap.dedent(
        f"""\
        User question: {query}

        Retrieved memories:
        {memory_context}
        """
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return (response.choices[0].message.content or "").strip()


def run_demo(
    *,
    query: str = DEFAULT_QUERY,
    techniques: list[str] | None = None,
    asker: str = "agent_b",
    use_llm: bool = True,
    model: str | None = None,
) -> None:
    names = techniques or list_techniques()
    print(f"\n{'=' * 72}")
    print(f"E2E memory demo (real backends) | task={DEFAULT_TASK} | asker={asker}")
    print(f"Query: {query}")
    print(f"{'=' * 72}\n")

    for technique in names:
        label = technique_label(technique)
        print(f"\n--- {label} ({technique}) ---\n")
        try:
            context, _hits, info = retrieve_for_technique(technique, query, asker=asker)
            print(f"Backend: {info.get('implementation', technique)}")
            print("Retrieved:")
            print(context)
        except ImportError as exc:
            print(f"Skipped: {exc}")
            continue
        except Exception as exc:
            print(f"Retrieval error: {exc}")
            continue

        if not use_llm:
            continue

        try:
            answer = answer_with_xai(query, context, technique=technique, model=model)
            print("\nLLM answer (xAI):")
            print(answer)
        except SystemExit as exc:
            print(f"\nLLM skipped: {exc}")
            break
        except Exception as exc:
            print(f"\nLLM error: {exc}")


def main(argv: list[str] | None = None) -> int:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(
        description="E2E demo with real mem0 / simple RAG / collaborative backends."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--asker", default="agent_b")
    parser.add_argument("--technique", action="append")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    reload_config()
    run_demo(
        query=args.query,
        techniques=args.technique,
        asker=args.asker,
        use_llm=not args.retrieval_only,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
