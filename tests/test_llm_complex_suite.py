"""Tests for complex LLM eval storyline and scoring."""

from __future__ import annotations

from evaluation.llm_complex_suite import (
    COMPLEX_LLM_EVAL_TASKS,
    COMPLEX_TASK_ID,
    seed_complex_storyline,
    task_success_score,
)
from evaluation.llm_fixtures import LLMEvalTask
from storage import init_db, session_scope
from storage.repository import MemoryRepository
from utils.config import AppConfig


def test_complex_suite_has_ten_tasks() -> None:
    assert len(COMPLEX_LLM_EVAL_TASKS) == 10


def test_seed_complex_storyline_creates_memories() -> None:
    cfg = AppConfig(database_url="sqlite:///:memory:", offline_retrieval=True)
    init_db(cfg)
    with session_scope(cfg) as session:
        repo = MemoryRepository(session)
        seed_complex_storyline(repo)
        rows = repo.list_by_task(COMPLEX_TASK_ID, include_archived=True)
    assert len(rows) >= 12
    contents = " ".join(m.content.lower() for m in rows)
    assert "oauth" in contents
    assert "pkce" in contents


def test_task_success_privacy() -> None:
    task = LLMEvalTask(
        id="x",
        query="q",
        asker="a",
        required_terms=(),
        forbidden_terms=("verbose",),
    )
    assert task_success_score("I do not have access to that information.", task) == 1.0
    assert task_success_score("agent_a uses verbose logs", task) == 0.0
