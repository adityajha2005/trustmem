"""Collaborative retrieval pipeline tests."""

from __future__ import annotations

from memory.schema import Memory, MemoryScope
from retrieval.context import RetrievalContext
from retrieval.retriever import build_retriever
from storage import init_db, session_scope
from storage.repository import MemoryRepository
from utils.config import AppConfig


def test_pipeline_excludes_archived_by_default() -> None:
    config = AppConfig(database_url="sqlite:///:memory:")
    init_db(config)
    with session_scope(config) as session:
        repo = MemoryRepository(session)
        archived = Memory(
            content="outdated deadline",
            agent="agent_a",
            scope=MemoryScope.SHARED,
            task_id="t1",
            archived=True,
        )
        active = Memory(
            content="confirmed deadline Friday",
            agent="agent_a",
            scope=MemoryScope.SHARED,
            task_id="t1",
        )
        repo.add(archived)
        repo.add(active)

        retriever = build_retriever(repo, config)
        ctx = RetrievalContext(agent_id="agent_b", task_id="t1")
        result = retriever.retrieve("deadline", ctx, top_k=5)

    assert result.filtered_archived == 1
    assert len(result.hits) == 1
    assert "Friday" in result.hits[0]["content"]
