"""Smoke tests for real memory technique backends."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from baselines.factory import create_backend
from memory.schema import Memory, MemoryScope
from retrieval.context import RetrievalContext
from storage import init_db, session_scope
from storage.repository import MemoryRepository
from utils.config import AppConfig


@pytest.fixture
def bench_config() -> AppConfig:
    return AppConfig(
        database_url="sqlite:///:memory:",
        offline_retrieval=True,
        faiss_index_path="./data/test.faiss",
    )


def test_simple_rag_backend_retrieves_shared_memory(bench_config: AppConfig) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        init_db(bench_config)
        with session_scope(bench_config) as session:
            repo = MemoryRepository(session)
            record = repo.add(
                Memory(
                    content="Deployment window is Tuesday 14:00 UTC.",
                    agent="agent_a",
                    scope=MemoryScope.SHARED,
                    task_id="t1",
                )
            )
            backend = create_backend(
                "simple_rag",
                repo,
                bench_config,
                work_dir=Path(tmp),
            )
            ctx = RetrievalContext(agent_id="agent_b", task_id="t1")
            result = backend.retrieve("deployment window", ctx, top_k=3)
            backend.close()
        ids = {h["memory_id"] for h in result.hits}
        assert str(record.id) in ids


def test_collaborative_backend_is_full_pipeline(bench_config: AppConfig) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        init_db(bench_config)
        with session_scope(bench_config) as session:
            repo = MemoryRepository(session)
            record = repo.add(
                Memory(
                    content="Redis cache TTL one hour for sessions.",
                    agent="agent_b",
                    scope=MemoryScope.SHARED,
                    task_id="t2",
                )
            )
            backend = create_backend(
                "collaborative",
                repo,
                bench_config,
                work_dir=Path(tmp),
            )
            assert backend.backend_info()["implementation"] == "collaborative_pipeline"
            ctx = RetrievalContext(agent_id="agent_a", task_id="t2")
            result = backend.retrieve("redis cache", ctx, top_k=3)
            backend.close()
        assert any(h["memory_id"] == str(record.id) for h in result.hits)


@pytest.mark.integration
def test_mem0_backend_user_scoped(bench_config: AppConfig) -> None:
    """Real Mem0 SDK; skipped in default pytest (loads torch). Run via technique benchmark."""
    if not os.getenv("MEM0_INTEGRATION_TEST"):
        pytest.skip("Set MEM0_INTEGRATION_TEST=1 to run real Mem0 tests")
    pytest.importorskip("mem0")

    with tempfile.TemporaryDirectory() as tmp:
        init_db(bench_config)
        with session_scope(bench_config) as session:
            repo = MemoryRepository(session)
            repo.add(
                Memory(
                    content="Agent A shared database pool size 50.",
                    agent="agent_a",
                    scope=MemoryScope.SHARED,
                    task_id="t3",
                )
            )
            own = repo.add(
                Memory(
                    content="Agent B private shortcut Ctrl-K.",
                    agent="agent_b",
                    scope=MemoryScope.LOCAL,
                    task_id="t3",
                )
            )
            backend = create_backend("mem0", repo, bench_config, work_dir=Path(tmp))
            ctx = RetrievalContext(agent_id="agent_b", task_id="t3")
            result = backend.retrieve("shortcut", ctx, top_k=5)
            backend.close()

        agents = {h.get("agent") for h in result.hits}
        assert agents <= {"agent_b"}
        assert any(h["memory_id"] == str(own.id) for h in result.hits)


@pytest.mark.parametrize(
    "technique",
    ["hybrid_bm25", "memgpt_tiered", "isolated_memory", "a_mem_linked"],
)
def test_literature_baselines_retrieve_shared(
    bench_config: AppConfig,
    technique: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        init_db(bench_config)
        with session_scope(bench_config) as session:
            repo = MemoryRepository(session)
            record = repo.add(
                Memory(
                    content="Redis cache TTL one hour for sessions.",
                    agent="agent_b",
                    scope=MemoryScope.SHARED,
                    task_id="t_lit",
                )
            )
            backend = create_backend(technique, repo, bench_config, work_dir=Path(tmp))
            ctx = RetrievalContext(agent_id="agent_a", task_id="t_lit")
            result = backend.retrieve("redis cache ttl", ctx, top_k=3)
            backend.close()
        if technique == "isolated_memory":
            assert not any(h["memory_id"] == str(record.id) for h in result.hits)
        else:
            assert any(h["memory_id"] == str(record.id) for h in result.hits)
