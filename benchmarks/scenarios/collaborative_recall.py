"""Collaborative recall benchmark: shared vs local scope correctness.

Three agents (A, B, C) populate a shared task with their own contributions.
Each agent then issues queries; retrieval must:
  - Return shared memories regardless of asker.
  - Return the asker's own local memories.
  - Never leak another agent's local memories.

Evaluates:
  - shared retrieval recall
  - scope correctness (no private leak across agents)
  - precision/recall over collaborative writes
"""

from __future__ import annotations

from benchmarks._helpers import BaseScenario, at, serialize_hits
from evaluation.harness import BenchmarkReport
from evaluation.metrics import mean, precision_at_k, recall_at_k
from memory.schema import Memory, MemoryScope
from retrieval.context import RetrievalContext
from storage import session_scope
from storage.sqlite_store import SQLiteMemoryStore


class CollaborativeRecallScenario(BaseScenario):
    name = "collaborative_recall_v1"
    expected = {
        "shared_recall_at_3": 1.0,
        "scope_correctness": 1.0,
        "no_private_leak": 1.0,
        "mean_precision_at_3": 0.4,
    }

    TASK_ID = "collab_recall"

    def run(self) -> BenchmarkReport:
        with session_scope(self.config) as session:
            store = SQLiteMemoryStore(session)
            retriever = self.build_retriever(session)

            # Agent A
            a_shared_db = store.create(
                Memory(
                    content="Database connection pool configured with 50 max connections",
                    agent="agent_a",
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.9,
                    importance=0.8,
                    timestamp=at(),
                )
            )
            a_local_secret = store.create(
                Memory(
                    content="Personal note: agent_a debugger profile uses verbose logs",
                    agent="agent_a",
                    scope=MemoryScope.LOCAL,
                    task_id=self.TASK_ID,
                    confidence=1.0,
                    importance=0.3,
                    timestamp=at(minutes=5),
                )
            )

            # Agent B
            b_shared_cache = store.create(
                Memory(
                    content="Cache layer uses Redis with one hour TTL for session data",
                    agent="agent_b",
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.85,
                    importance=0.7,
                    timestamp=at(minutes=10),
                )
            )
            b_local_note = store.create(
                Memory(
                    content="Personal note: agent_b prefers shortcut Ctrl-K for navigation",
                    agent="agent_b",
                    scope=MemoryScope.LOCAL,
                    task_id=self.TASK_ID,
                    confidence=1.0,
                    importance=0.2,
                    timestamp=at(minutes=15),
                )
            )

            # Agent C
            c_shared_queue = store.create(
                Memory(
                    content="Message queue uses RabbitMQ with priority routing for tasks",
                    agent="agent_c",
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.9,
                    importance=0.8,
                    timestamp=at(minutes=20),
                )
            )

            forbidden_for = {
                "agent_a": {str(b_local_note.id)},
                "agent_b": {str(a_local_secret.id)},
                "agent_c": {str(a_local_secret.id), str(b_local_note.id)},
            }

            # (asker, query, relevant_ids)
            test_queries = [
                ("agent_b", "database connection pool", {str(a_shared_db.id)}),
                ("agent_b", "redis cache session ttl", {str(b_shared_cache.id)}),
                ("agent_b", "personal note shortcut Ctrl", {str(b_local_note.id)}),
                ("agent_c", "rabbitmq priority routing tasks", {str(c_shared_queue.id)}),
                ("agent_c", "agent_b shortcut personal", set()),  # must not leak
                ("agent_a", "debugger verbose logs personal", {str(a_local_secret.id)}),
            ]

            traces: list[dict] = []
            shared_recalls: list[float] = []
            precisions: list[float] = []
            scope_correct: list[float] = []
            leak_count = 0
            total_retrieved = 0

            for asker, query, relevant in test_queries:
                ctx = RetrievalContext(agent_id=asker, task_id=self.TASK_ID)
                result = retriever.retrieve(query, ctx, top_k=3)
                retrieved = [h["memory_id"] for h in result.hits]
                total_retrieved += len(retrieved)

                forbidden = forbidden_for.get(asker, set())
                leaks = [r for r in retrieved if r in forbidden]
                leak_count += len(leaks)
                scope_correct.append(0.0 if leaks else 1.0)

                p3 = precision_at_k(relevant, retrieved, k=3)
                r3 = recall_at_k(relevant, retrieved, k=3) if relevant else 1.0

                if relevant:
                    shared_recalls.append(r3)
                precisions.append(p3 if relevant else 1.0)

                traces.append(
                    self.enrich_trace(
                        {
                            "asker": asker,
                            "query": query,
                            "relevant_ids": sorted(relevant),
                            "retrieved_ids": retrieved,
                            "leaks": leaks,
                            "precision_at_3": p3,
                            "recall_at_3": r3,
                            "scope_correct": not leaks,
                            "hits": serialize_hits(result.hits),
                        },
                        result,
                    )
                )

            metrics = {
                "shared_recall_at_3": mean(shared_recalls),
                "scope_correctness": mean(scope_correct),
                "no_private_leak": 1.0 if leak_count == 0 else 0.0,
                "private_leak_rate": (leak_count / total_retrieved) if total_retrieved else 0.0,
                "mean_precision_at_3": mean(precisions),
                "mean_recall_at_3": mean(shared_recalls),
            }

            return BenchmarkReport(
                scenario=self.name,
                metrics=metrics,
                expected=self.expected,
                queries=traces,
                notes={
                    "agents": ["agent_a", "agent_b", "agent_c"],
                    "shared_count": 3,
                    "local_count": 2,
                },
            )
