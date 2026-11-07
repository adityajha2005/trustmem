"""Memory evolution workflow tests."""

from __future__ import annotations

from memory.evolution import MemoryEvolution
from memory.schema import Memory, MemoryScope, MemoryType
from storage import init_db, session_scope
from storage.sqlite_store import SQLiteMemoryStore
from utils.config import AppConfig


def test_planner_coder_verifier_workflow() -> None:
    config = AppConfig(database_url="sqlite:///:memory:")
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        evolution = MemoryEvolution(store)

        plan = store.create(
            Memory(
                content="Implement feature X with REST",
                agent="planner",
                scope=MemoryScope.SHARED,
                task_id="task_1",
                memory_type=MemoryType.PLAN,
            )
        )
        impl = evolution.supersede(
            plan.id,
            Memory(
                content="Feature X implemented with REST endpoints",
                agent="coder",
                scope=MemoryScope.SHARED,
                task_id="task_1",
                memory_type=MemoryType.OBSERVATION,
            ),
        )
        wrong = store.create(
            Memory(
                content="Feature X uses gRPC only",
                agent="coder",
                scope=MemoryScope.SHARED,
                task_id="task_1",
                confidence=0.6,
            )
        )
        claim = evolution.register_contradiction(
            [impl.id, wrong.id],
            detected_by="verifier",
            notes="Transport mismatch",
        )
        evolution.resolve_conflicts(claim.id, winner_id=impl.id, requesting_agent="verifier")

        active = evolution.get_active_memories("task_1")
        active_ids = {m.id for m in active}
        assert impl.id in active_ids
        assert wrong.id not in active_ids
        assert plan.id not in active_ids

        latest = evolution.get_latest_memory(plan.id)
        assert latest is not None
        assert latest.id == impl.id
