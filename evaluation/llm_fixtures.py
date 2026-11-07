"""Shared memory fixtures and task definitions for LLM evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.schema import Memory, MemoryScope
from storage.repository import MemoryRepository

LLM_EVAL_TASK_ID = "llm_eval_v1"


@dataclass(frozen=True)
class LLMEvalTask:
    """One question with rubric for automatic + LLM-judge scoring."""

    id: str
    query: str
    asker: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...] = ()
    description: str = ""


# Two tasks designed to stress multi-agent shared memory vs user-scoped (Mem0).
LLM_EVAL_TASKS: tuple[LLMEvalTask, ...] = (
    LLMEvalTask(
        id="cross_agent_db_pool",
        query="What database connection pool configuration did the team set for production?",
        asker="agent_b",
        required_terms=("50", "connection", "pool"),
        forbidden_terms=("ctrl-k", "shortcut", "verbose logs"),
        description=(
            "Agent A wrote the shared DB fact; agent B asks. "
            "Collaborative/simple RAG should retrieve it; Mem0 (per-user) may miss."
        ),
    ),
    LLMEvalTask(
        id="team_infra_cache_queue",
        query="What cache and message queue does the team use for sessions and background tasks?",
        asker="agent_b",
        required_terms=("redis", "rabbitmq"),
        forbidden_terms=("ctrl-k", "debugger", "verbose"),
        description=(
            "Requires shared facts from agent B (Redis) and agent C (RabbitMQ). "
            "Tests cross-agent shared recall, not only agent B local notes."
        ),
    ),
)


def seed_llm_eval_memories(repo: MemoryRepository) -> None:
    """Deterministic multi-agent memory store for all techniques."""
    facts: list[tuple[str, MemoryScope, str, float]] = [
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
                task_id=LLM_EVAL_TASK_ID,
                confidence=confidence,
                importance=0.75,
            )
        )


def term_recall(terms: tuple[str, ...], text: str) -> float:
    """Fraction of required terms present in text (case-insensitive)."""
    if not terms:
        return 1.0
    lower = text.lower()
    hit = sum(1 for t in terms if t.lower() in lower)
    return hit / len(terms)


def has_forbidden(terms: tuple[str, ...], text: str) -> bool:
    lower = text.lower()
    return any(t.lower() in lower for t in terms)
