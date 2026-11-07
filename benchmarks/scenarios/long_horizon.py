"""Long-horizon benchmark: multi-session evolving plan for a feature.

Simulates four sessions across days that gradually supersede earlier
requirements while the verifier confirms the final state. Retrieval should
prefer the latest active state at every query, with low stale rate.

Evaluates:
  - long-horizon consistency (mean active accuracy across queries)
  - stale retrieval rate
  - context efficiency (useful retrieved per top-k slot)
"""

from __future__ import annotations

from benchmarks._helpers import BaseScenario, at, serialize_hits
from evaluation.harness import BenchmarkReport
from evaluation.metrics import (
    active_memory_accuracy,
    context_efficiency,
    long_horizon_consistency,
    mean,
    precision_at_k,
    recall_at_k,
    stale_retrieval_rate,
)
from memory.evolution import MemoryEvolution
from memory.schema import Memory, MemoryScope, MemoryType
from retrieval.context import RetrievalContext
from storage import session_scope
from storage.sqlite_store import SQLiteMemoryStore


class LongHorizonScenario(BaseScenario):
    name = "long_horizon_v1"
    expected = {
        "long_horizon_consistency": 0.65,
        "active_memory_accuracy": 0.65,
        "stale_retrieval_rate": ("<=", 0.25),
        "mean_recall_at_3": 0.65,
    }

    TASK_ID = "feature_multitenant"

    def run(self) -> BenchmarkReport:
        with session_scope(self.config) as session:
            store = SQLiteMemoryStore(session)
            evolution = MemoryEvolution(store)
            retriever = self.build_retriever(session)

            # ------------------------------------------------------------------
            # Session 1 (day 0): planner sets initial requirements
            # ------------------------------------------------------------------
            req_001 = store.create(
                Memory(
                    content="REQ-001: System must support multi-tenant data isolation",
                    agent="planner",
                    memory_type=MemoryType.PLAN,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.9,
                    importance=0.9,
                    timestamp=at(),
                )
            )
            req_002_v1 = store.create(
                Memory(
                    content="REQ-002: Use shared database with tenant_id column for isolation",
                    agent="planner",
                    memory_type=MemoryType.PLAN,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.7,
                    importance=0.7,
                    timestamp=at(minutes=10),
                )
            )
            req_003_v1 = store.create(
                Memory(
                    content="REQ-003: Implement row-level security via PostgreSQL RLS policies",
                    agent="planner",
                    memory_type=MemoryType.PLAN,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.7,
                    importance=0.7,
                    timestamp=at(minutes=20),
                )
            )

            # ------------------------------------------------------------------
            # Session 2 (day 1): architect supersedes REQ-002
            # ------------------------------------------------------------------
            req_002_v2 = evolution.supersede(
                req_002_v1.id,
                Memory(
                    content="REQ-002: Use schema-per-tenant isolation for stronger boundaries",
                    agent="architect",
                    memory_type=MemoryType.PLAN,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.85,
                    importance=0.85,
                    timestamp=at(days=1),
                ),
            )

            # ------------------------------------------------------------------
            # Session 3 (day 3): coder supersedes REQ-003
            # ------------------------------------------------------------------
            req_003_v2 = evolution.supersede(
                req_003_v1.id,
                Memory(
                    content="REQ-003: Implement schema isolation via PostgreSQL search_path",
                    agent="coder",
                    memory_type=MemoryType.OBSERVATION,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.9,
                    importance=0.85,
                    timestamp=at(days=3),
                ),
            )

            # ------------------------------------------------------------------
            # Session 4 (day 5): verifier confirms current state
            # ------------------------------------------------------------------
            verification = store.create(
                Memory(
                    content="Verified schema-per-tenant works under concurrent connections",
                    agent="verifier",
                    memory_type=MemoryType.OBSERVATION,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.95,
                    importance=0.9,
                    timestamp=at(days=5),
                )
            )

            active_ids = {
                str(req_001.id),
                str(req_002_v2.id),
                str(req_003_v2.id),
                str(verification.id),
            }
            stale_ids = {str(req_002_v1.id), str(req_003_v1.id)}

            # Each query targets a topic the active set should answer
            test_queries = [
                ("multi-tenant data isolation requirements", {str(req_001.id)}),
                ("schema per tenant isolation", {str(req_002_v2.id)}),
                ("schema isolation search_path implementation", {str(req_003_v2.id)}),
                ("verified concurrent tenant isolation", {str(verification.id)}),
            ]

            ctx = RetrievalContext(agent_id="reader", task_id=self.TASK_ID)
            traces: list[dict] = []
            active_accs: list[float] = []
            stale_rates: list[float] = []
            efficiencies: list[float] = []
            recalls: list[float] = []
            precisions: list[float] = []

            for query, relevant in test_queries:
                result = retriever.retrieve(query, ctx, top_k=4)
                retrieved = [h["memory_id"] for h in result.hits]

                active_acc = active_memory_accuracy(retrieved, active_ids)
                stale = stale_retrieval_rate(retrieved, stale_ids)
                eff = context_efficiency(retrieved, relevant, k=4)
                p3 = precision_at_k(relevant, retrieved, k=3)
                r3 = recall_at_k(relevant, retrieved, k=3)

                active_accs.append(active_acc)
                stale_rates.append(stale)
                efficiencies.append(eff)
                precisions.append(p3)
                recalls.append(r3)

                traces.append(
                    self.enrich_trace(
                        {
                            "query": query,
                            "relevant_ids": list(relevant),
                            "retrieved_ids": retrieved,
                            "active_accuracy": active_acc,
                            "stale_rate": stale,
                            "context_efficiency": eff,
                            "precision_at_3": p3,
                            "recall_at_3": r3,
                            "filtered_lineage": result.filtered_lineage,
                            "filtered_archived": result.filtered_archived,
                            "hits": serialize_hits(result.hits),
                        },
                        result,
                    )
                )

            metrics = {
                "long_horizon_consistency": long_horizon_consistency(active_accs),
                "active_memory_accuracy": mean(active_accs),
                "stale_retrieval_rate": mean(stale_rates),
                "context_efficiency": mean(efficiencies),
                "mean_precision_at_3": mean(precisions),
                "mean_recall_at_3": mean(recalls),
            }

            return BenchmarkReport(
                scenario=self.name,
                metrics=metrics,
                expected=self.expected,
                queries=traces,
                notes={
                    "active_ids": sorted(active_ids),
                    "stale_ids": sorted(stale_ids),
                    "sessions": 4,
                },
            )
