"""Minimal scenario: multi-agent writes in one task, cross-agent recall."""

from __future__ import annotations

from memory.schema import Memory, MemoryScope
from evaluation.metrics import CollaborativeMetrics, recall_at_k
from benchmarks.protocol import CollaborativeScenario
from storage import init_db, session_scope
from storage.repository import MemoryRepository
from retrieval.context import RetrievalContext
from retrieval.retriever import build_retriever
from utils.config import AppConfig


class SessionRecallScenario(CollaborativeScenario):
    """
    Agent A writes a shared fact; Agent B must retrieve it.

    Lightweight, deterministic, no LLM — suitable for CI smoke tests.
    """

    name = "session_recall_v1"

    def __init__(self, database_url: str = "sqlite:///:memory:") -> None:
        self._database_url = database_url
        self._memory_id: str | None = None

    def setup(self) -> None:
        self._config = AppConfig(database_url=self._database_url)
        init_db(self._config)

    def run(self) -> CollaborativeMetrics:
        with session_scope(self._config) as session:
            repo = MemoryRepository(session)
            record = Memory(
                content="The deployment window is Tuesday 14:00 UTC.",
                agent="agent_a",
                scope=MemoryScope.SHARED,
                task_id="bench_task_1",
                confidence=0.95,
                importance=0.8,
            )
            repo.add(record)
            self._memory_id = str(record.id)

            retriever = build_retriever(repo, self._config)
            ctx = RetrievalContext(agent_id="agent_b", task_id="bench_task_1")
            result = retriever.retrieve("deployment window", ctx, top_k=3)

        retrieved_ids = [h["memory_id"] for h in result.hits]
        r_at_k = recall_at_k({self._memory_id}, retrieved_ids, k=3)
        return CollaborativeMetrics(recall_at_k=r_at_k, cross_agent_consistency=1.0 if r_at_k else 0.0)

    def teardown(self) -> None:
        pass
