"""Complex multi-agent memory storyline + 10 LLM evaluation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from benchmarks._helpers import at
from evaluation.llm_fixtures import LLMEvalTask, has_forbidden, term_recall
from memory.evolution import MemoryEvolution
from memory.schema import Memory, MemoryScope, MemoryType
from storage.repository import MemoryRepository

COMPLEX_TASK_ID = "llm_eval_complex_v1"


@dataclass(frozen=True)
class EvalSuite:
    name: str
    task_id: str
    tasks: tuple[LLMEvalTask, ...]
    seed_fn: Callable[[MemoryRepository], None]


def seed_complex_storyline(repo: MemoryRepository) -> None:
    """
    Rich storyline: auth migration, trust conflicts, observability, infra, privacy.

    Mirrors benchmark scenarios (contradiction, trust, long horizon) in one task_id.
    """
    store = repo.store
    evolution = MemoryEvolution(store)

    plan = store.create(
        Memory(
            content="Architecture decision: use JWT for authentication tokens",
            agent="planner",
            memory_type=MemoryType.PLAN,
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
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
            task_id=COMPLEX_TASK_ID,
            confidence=0.85,
            importance=0.85,
            timestamp=at(hours=2),
        ),
    )
    store.create(
        Memory(
            content="JWT implementation has stateless logout vulnerability",
            agent="verifier",
            memory_type=MemoryType.REFLECTION,
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
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
            task_id=COMPLEX_TASK_ID,
            confidence=0.95,
            importance=0.95,
            timestamp=at(hours=12),
        ),
    )
    claim = evolution.register_contradiction(
        [jwt_impl.id, oauth_impl.id],
        detected_by="verifier",
        confidence=0.95,
        notes="JWT vs OAuth auth conflict",
    )
    evolution.resolve_conflicts(
        claim.id,
        requesting_agent="verifier",
        task_id=COMPLEX_TASK_ID,
    )

    store.create(
        Memory(
            content="Deployment occurs at 14:00 UTC every Friday afternoon",
            agent="planner",
            memory_type=MemoryType.PLAN,
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.6,
            importance=0.8,
            timestamp=at(hours=1),
        )
    )
    store.create(
        Memory(
            content="Deployment scheduled at 14:00 UTC on Friday afternoon",
            agent="coder",
            memory_type=MemoryType.OBSERVATION,
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.7,
            importance=0.8,
            timestamp=at(hours=1, minutes=1),
        )
    )
    store.create(
        Memory(
            content="Verified: deployment now occurs at 09:00 UTC every Wednesday",
            agent="verifier",
            memory_type=MemoryType.REFLECTION,
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.95,
            importance=0.9,
            timestamp=at(hours=1, minutes=2),
        )
    )

    store.create(
        Memory(
            content="Database connection pool configured with 50 max connections for production",
            agent="agent_a",
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.92,
            importance=0.85,
            timestamp=at(hours=3),
        )
    )
    store.create(
        Memory(
            content="Cache layer uses Redis with one hour TTL for session data",
            agent="agent_b",
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.88,
            importance=0.8,
            timestamp=at(hours=4),
        )
    )
    store.create(
        Memory(
            content="Message queue uses RabbitMQ with priority routing for background tasks",
            agent="agent_c",
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.9,
            importance=0.8,
            timestamp=at(hours=5),
        )
    )
    store.create(
        Memory(
            content="API latency SLO: p99 must stay under 200 milliseconds",
            agent="agent_d",
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.9,
            importance=0.85,
            timestamp=at(hours=8),
        )
    )
    rollback_old = store.create(
        Memory(
            content="Rollback procedure: revert to release v2.3.1 using blue-green switch",
            agent="agent_b",
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.7,
            importance=0.6,
            timestamp=at(hours=9),
        )
    )
    evolution.supersede(
        rollback_old.id,
        Memory(
            content="Updated rollback procedure: canary rollback with 5% traffic slice",
            agent="verifier",
            scope=MemoryScope.SHARED,
            task_id=COMPLEX_TASK_ID,
            confidence=0.92,
            importance=0.75,
            timestamp=at(hours=15),
        ),
    )

    store.create(
        Memory(
            content="Personal note: agent_a debugger profile uses verbose logs",
            agent="agent_a",
            scope=MemoryScope.LOCAL,
            task_id=COMPLEX_TASK_ID,
            confidence=1.0,
            importance=0.3,
            timestamp=at(hours=10),
        )
    )
    store.create(
        Memory(
            content="Personal note: agent_b prefers shortcut Ctrl-K for navigation",
            agent="agent_b",
            scope=MemoryScope.LOCAL,
            task_id=COMPLEX_TASK_ID,
            confidence=1.0,
            importance=0.2,
            timestamp=at(hours=11),
        )
    )


COMPLEX_LLM_EVAL_TASKS: tuple[LLMEvalTask, ...] = (
    LLMEvalTask(
        id="auth_current_after_migration",
        query=(
            "After the security migration, what authentication mechanism is in production "
            "today — include the flow name?"
        ),
        asker="reader",
        required_terms=("oauth", "pkce"),
        forbidden_terms=("jwt", "hs256"),
        description="Contradiction resolution: OAuth winner, JWT archived.",
    ),
    LLMEvalTask(
        id="deployment_verified_schedule",
        query=(
            "What is the verified production deployment schedule (day and UTC time)? "
            "Ignore earlier planner/coder drafts."
        ),
        asker="reader",
        required_terms=("wednesday", "09:00"),
        forbidden_terms=("friday", "14:00"),
        description="Trust calibration: verifier overrides planner/coder.",
    ),
    LLMEvalTask(
        id="multi_hop_capacity_planning",
        query=(
            "For the upcoming load test, summarize database pool capacity AND "
            "how background work is routed in the message queue."
        ),
        asker="agent_b",
        required_terms=("50", "rabbitmq", "priority"),
        forbidden_terms=("ctrl-k",),
        description="Requires agent_a DB + agent_c queue shared facts.",
    ),
    LLMEvalTask(
        id="observability_slo",
        query="What p99 latency SLO did the team set for the API?",
        asker="agent_c",
        required_terms=("200", "p99", "latency"),
        forbidden_terms=(),
        description="Single-hop from agent_d observability memory.",
    ),
    LLMEvalTask(
        id="rollback_active_procedure",
        query="What is the current rollback procedure after the latest verifier update?",
        asker="agent_b",
        required_terms=("canary", "5%"),
        forbidden_terms=("v2.3.1", "blue-green"),
        description="Lineage: superseded rollback must not surface.",
    ),
    LLMEvalTask(
        id="privacy_agent_a_local",
        query="What debugger logging preference does agent_a use personally?",
        asker="agent_c",
        required_terms=(),
        forbidden_terms=("verbose", "debugger", "agent_a"),
        description="Scope: agent_c must NOT read agent_a local memory.",
    ),
    LLMEvalTask(
        id="cross_agent_redis_ttl",
        query="As agent_a, what TTL does our shared session cache use?",
        asker="agent_a",
        required_terms=("redis", "hour", "ttl"),
        forbidden_terms=(),
        description="agent_a asks; fact authored by agent_b in shared pool.",
    ),
    LLMEvalTask(
        id="stale_jwt_suppression",
        query="Describe the active JWT token signing algorithm in production.",
        asker="reader",
        required_terms=(),
        forbidden_terms=("hs256",),
        description="Stale/archived JWT must not be presented as current.",
    ),
    LLMEvalTask(
        id="infra_cache_and_db",
        query=(
            "List the production cache technology with TTL policy and the database "
            "pool max connection count."
        ),
        asker="reader",
        required_terms=("redis", "50"),
        forbidden_terms=(),
        description="Two shared infra facts in one answer.",
    ),
    LLMEvalTask(
        id="contradiction_winner_rationale",
        query=(
            "Why did the team move away from JWT — what vulnerability was flagged and "
            "what replaced it?"
        ),
        asker="reader",
        required_terms=("logout", "oauth", "pkce"),
        forbidden_terms=(),
        description="Needs verifier finding + OAuth resolution.",
    ),
)


def get_suite(name: str) -> EvalSuite:
    from evaluation.llm_fixtures import LLM_EVAL_TASKS, LLM_EVAL_TASK_ID, seed_llm_eval_memories

    if name == "basic":
        return EvalSuite(
            name="basic",
            task_id=LLM_EVAL_TASK_ID,
            tasks=LLM_EVAL_TASKS,
            seed_fn=seed_llm_eval_memories,
        )
    if name == "complex":
        return EvalSuite(
            name="complex",
            task_id=COMPLEX_TASK_ID,
            tasks=COMPLEX_LLM_EVAL_TASKS,
            seed_fn=seed_complex_storyline,
        )
    raise KeyError(f"Unknown suite {name!r}; use 'basic' or 'complex'")


def task_success_score(answer: str, task: LLMEvalTask) -> float:
    """Factual success: required terms present and no forbidden leaks."""
    if has_forbidden(task.forbidden_terms, answer):
        return 0.0
    if not task.required_terms:
        return 1.0
    return term_recall(task.required_terms, answer)
