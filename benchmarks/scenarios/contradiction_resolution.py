"""Contradiction-resolution benchmark: JWT → OAuth migration.

Workflow:
  1. Planner defines auth architecture (JWT).
  2. Coder implements JWT.
  3. Verifier flags a security issue.
  4. Coder migrates the implementation to OAuth.
  5. Verifier registers a contradiction; resolution must elect OAuth.

Evaluates:
  - stale memory suppression (archived JWT layers)
  - active memory retrieval (OAuth + verifier finding)
  - lineage correctness (plan → JWT → OAuth)
  - contradiction resolution (winner = OAuth)
"""

from __future__ import annotations

from benchmarks._helpers import BaseScenario, at, serialize_hits
from evaluation.harness import BenchmarkReport
from evaluation.metrics import (
    active_memory_accuracy,
    precision_at_k,
    stale_retrieval_rate,
)
from memory.evolution import MemoryEvolution
from memory.schema import Memory, MemoryScope, MemoryType
from retrieval.context import RetrievalContext
from storage import session_scope
from storage.sqlite_store import SQLiteMemoryStore


class ContradictionResolutionScenario(BaseScenario):
    name = "contradiction_resolution_v1"
    expected = {
        "active_memory_accuracy": 1.0,
        "stale_retrieval_rate": ("<=", 0.0),
        "lineage_completeness": 1.0,
        "contradiction_resolution_rate": 1.0,
        "precision_at_1": 1.0,
    }

    TASK_ID = "auth_migration"

    def run(self) -> BenchmarkReport:
        with session_scope(self.config) as session:
            store = SQLiteMemoryStore(session)
            evolution = MemoryEvolution(store)

            plan = store.create(
                Memory(
                    content="Architecture decision: use JWT for authentication tokens",
                    agent="planner",
                    memory_type=MemoryType.PLAN,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.8,
                    importance=0.9,
                    timestamp=at(),
                )
            )
            jwt_impl = evolution.supersede(
                plan.id,
                Memory(
                    content="Implemented JWT authentication with HS256 signed tokens",
                    agent="coder",
                    memory_type=MemoryType.OBSERVATION,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.85,
                    importance=0.85,
                    timestamp=at(hours=2),
                ),
            )
            verifier_finding = store.create(
                Memory(
                    content="JWT implementation has stateless logout vulnerability",
                    agent="verifier",
                    memory_type=MemoryType.REFLECTION,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.95,
                    importance=1.0,
                    timestamp=at(hours=6),
                )
            )
            oauth_impl = evolution.supersede(
                jwt_impl.id,
                Memory(
                    content="Migrated authentication to OAuth 2.0 with PKCE flow",
                    agent="coder",
                    memory_type=MemoryType.OBSERVATION,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.95,
                    importance=0.95,
                    timestamp=at(hours=12),
                ),
            )

            # Lineage check before contradiction resolution
            lineage_chain = evolution.get_memory_lineage(plan.id, task_id=self.TASK_ID)
            actual_chain_ids = [m.id for m in lineage_chain]
            expected_chain_ids = [plan.id, jwt_impl.id, oauth_impl.id]
            lineage_ok = actual_chain_ids == expected_chain_ids

            # Verifier registers a contradiction between JWT and OAuth implementations
            claim = evolution.register_contradiction(
                [jwt_impl.id, oauth_impl.id],
                detected_by="verifier",
                confidence=0.95,
                notes="Auth method conflict: JWT vs OAuth",
            )

            retriever = self.build_retriever(session, registry=evolution.registry)
            ctx = RetrievalContext(agent_id="reader", task_id=self.TASK_ID)
            result = retriever.retrieve("authentication oauth migration", ctx, top_k=5)
            retrieved_ids = [h["memory_id"] for h in result.hits]

            active_ids = {str(oauth_impl.id), str(verifier_finding.id)}
            stale_ids = {str(plan.id), str(jwt_impl.id)}

            # Resolve the contradiction (do not double-archive — supersede already did)
            resolved = evolution.resolve_conflicts(
                claim.id,
                requesting_agent="verifier",
                archive_losers=False,
                task_id=self.TASK_ID,
            )
            resolution_ok = resolved.winner_id == oauth_impl.id

            metrics = {
                "active_memory_accuracy": active_memory_accuracy(retrieved_ids, active_ids),
                "stale_retrieval_rate": stale_retrieval_rate(retrieved_ids, stale_ids),
                "precision_at_1": precision_at_k(active_ids, retrieved_ids, k=1),
                "lineage_completeness": 1.0 if lineage_ok else 0.0,
                "contradiction_resolution_rate": 1.0 if resolution_ok else 0.0,
                "filtered_archived": float(result.filtered_archived),
                "filtered_lineage": float(result.filtered_lineage),
            }

            return BenchmarkReport(
                scenario=self.name,
                metrics=metrics,
                expected=self.expected,
                queries=[
                    self.enrich_trace(
                        {
                            "query": result.query,
                            "agent_id": ctx.agent_id,
                            "top_k": 5,
                            "retrieved_ids": retrieved_ids,
                            "hits": serialize_hits(result.hits),
                        },
                        result,
                    )
                ],
                notes={
                    "expected_chain": [str(x) for x in expected_chain_ids],
                    "actual_chain": [str(x) for x in actual_chain_ids],
                    "winner_id": str(resolved.winner_id),
                    "expected_winner": str(oauth_impl.id),
                },
            )
