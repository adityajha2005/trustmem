"""Trust-calibration benchmark: confidence-aware ranking + verifier override.

Three agents make conflicting claims about a deployment schedule with
distinct confidence levels. The benchmark probes:
  - Pre-resolution: trust-aware ranking should already place the highest
    confidence claim first (no inversions).
  - Post-resolution: contradiction registry must elect the verifier and
    suppress losing memories from retrieval.

Evaluates:
  - ranking_inversion_rate (lower is better)
  - precision@1 (verifier wins top-1)
  - contradiction_suppression_rate (losers no longer surface)
  - resolution correctness
"""

from __future__ import annotations

from datetime import timedelta

from benchmarks._helpers import BaseScenario, at, serialize_hits
from evaluation.harness import BenchmarkReport
from evaluation.metrics import (
    precision_at_k,
    ranking_inversion_rate,
    rerank_disruption_rate,
    suppression_rate,
    trust_preservation_rate,
)
from memory.evolution import MemoryEvolution
from memory.schema import Memory, MemoryScope, MemoryType
from retrieval.context import RetrievalContext
from storage import session_scope
from storage.sqlite_store import SQLiteMemoryStore


class TrustCalibrationScenario(BaseScenario):
    name = "trust_calibration_v1"
    expected = {
        "precision_at_1": 1.0,
        "contradiction_suppression_rate": 1.0,
        "resolution_correct": 1.0,
        "ranking_inversion_rate": ("<=", 0.0),
    }

    TASK_ID = "trust_eval"

    def run(self) -> BenchmarkReport:
        with session_scope(self.config) as session:
            store = SQLiteMemoryStore(session)
            evolution = MemoryEvolution(store)
            retriever = self.build_retriever(session, registry=evolution.registry)

            base_t = at()
            planner_claim = store.create(
                Memory(
                    content="Deployment occurs at 14:00 UTC every Friday afternoon",
                    agent="planner",
                    memory_type=MemoryType.PLAN,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.6,
                    importance=0.8,
                    timestamp=base_t,
                )
            )
            coder_claim = store.create(
                Memory(
                    content="Deployment scheduled at 14:00 UTC on Friday afternoon",
                    agent="coder",
                    memory_type=MemoryType.OBSERVATION,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.7,
                    importance=0.8,
                    timestamp=base_t + timedelta(seconds=1),
                )
            )
            verifier_claim = store.create(
                Memory(
                    content="Verified: deployment now occurs at 09:00 UTC every Wednesday",
                    agent="verifier",
                    memory_type=MemoryType.REFLECTION,
                    scope=MemoryScope.SHARED,
                    task_id=self.TASK_ID,
                    confidence=0.95,
                    importance=0.9,
                    timestamp=base_t + timedelta(seconds=2),
                )
            )

            confidence_by_id = {
                str(planner_claim.id): 0.6,
                str(coder_claim.id): 0.7,
                str(verifier_claim.id): 0.95,
            }
            losers = {str(planner_claim.id), str(coder_claim.id)}

            ctx = RetrievalContext(agent_id="reader", task_id=self.TASK_ID)

            # Phase 1: pre-resolution — trust-aware ranking should hold
            pre = retriever.retrieve("deployment time schedule", ctx, top_k=3)
            pre_ids = [h["memory_id"] for h in pre.hits]
            inversion_rate = ranking_inversion_rate(pre_ids, confidence_by_id)
            p_at_1_pre = precision_at_k({str(verifier_claim.id)}, pre_ids, k=1)

            # Phase 2: register and resolve contradiction (archives losers)
            claim = evolution.register_contradiction(
                [planner_claim.id, coder_claim.id, verifier_claim.id],
                detected_by="verifier",
                confidence=0.95,
                notes="Schedule conflict; verifier overrides",
            )
            resolved = evolution.resolve_conflicts(
                claim.id,
                requesting_agent="verifier",
                task_id=self.TASK_ID,
            )

            # Phase 3: post-resolution — losers must be suppressed
            post = retriever.retrieve("deployment time schedule", ctx, top_k=3)
            post_ids = [h["memory_id"] for h in post.hits]
            suppressed = sum(1 for lid in losers if lid not in post_ids)

            # Trust preservation + rerank disruption (fusion-stage diagnostics).
            pre_pre_rerank_ids = [
                h["memory_id"]
                for h in sorted(
                    pre.hits,
                    key=lambda h: h.get("pre_rerank_rank", 0),
                )
            ]
            preservation = trust_preservation_rate(
                pre_ids,
                confidence_by_id,
                k=1,
            )
            disruption = rerank_disruption_rate(pre_pre_rerank_ids, pre_ids, k=3)

            metrics = {
                "ranking_inversion_rate": inversion_rate,
                "calibration_quality": 1.0 - inversion_rate,
                "precision_at_1": p_at_1_pre,
                "precision_at_1_post": precision_at_k(
                    {str(verifier_claim.id)}, post_ids, k=1
                ),
                "contradiction_suppression_rate": suppression_rate(suppressed, len(losers)),
                "resolution_correct": 1.0 if resolved.winner_id == verifier_claim.id else 0.0,
                "trust_preservation_rate": preservation,
                "rerank_disruption_rate": disruption,
            }

            return BenchmarkReport(
                scenario=self.name,
                metrics=metrics,
                expected=self.expected,
                queries=[
                    self.enrich_trace(
                        {
                            "phase": "pre_resolution",
                            "query": pre.query,
                            "retrieved_ids": pre_ids,
                            "hits": serialize_hits(pre.hits),
                        },
                        pre,
                    ),
                    self.enrich_trace(
                        {
                            "phase": "post_resolution",
                            "query": post.query,
                            "retrieved_ids": post_ids,
                            "hits": serialize_hits(post.hits),
                        },
                        post,
                    ),
                ],
                notes={
                    "expected_winner": str(verifier_claim.id),
                    "winner_id": str(resolved.winner_id),
                    "confidence_by_id": confidence_by_id,
                },
            )
