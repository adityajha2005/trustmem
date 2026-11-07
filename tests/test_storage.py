"""Storage layer tests."""

from __future__ import annotations

import pytest

from memory.schema import Memory, MemoryScope, MemoryType, utc_now
from storage import init_db, session_scope
from storage.sqlite_store import SQLiteMemoryStore
from utils.config import AppConfig


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(database_url="sqlite:///:memory:")


def test_create_and_get_local_memory(config: AppConfig) -> None:
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        memory = Memory(
            content="private note",
            agent="agent_a",
            scope=MemoryScope.LOCAL,
            task_id="task_1",
        )
        created = store.create(memory)
        loaded = store.get(created.id)

    assert loaded is not None
    assert loaded.content == "private note"
    assert loaded.scope == MemoryScope.LOCAL
    assert loaded.timestamp.tzinfo is not None


def test_shared_memory_visible_via_list_for_agent(config: AppConfig) -> None:
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        store.create(
            Memory(
                content="team fact",
                agent="agent_a",
                scope=MemoryScope.SHARED,
                task_id="task_1",
            )
        )
        store.create(
            Memory(
                content="private",
                agent="agent_b",
                scope=MemoryScope.LOCAL,
                task_id="task_1",
            )
        )
        visible = store.list_for_agent("agent_b", task_id="task_1")

    assert len(visible) == 2


def test_local_memory_hidden_from_other_agents(config: AppConfig) -> None:
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        store.create(
            Memory(
                content="secret",
                agent="agent_a",
                scope=MemoryScope.LOCAL,
                task_id="task_1",
            )
        )
        visible = store.list_for_agent("agent_b", task_id="task_1")

    assert len(visible) == 0


def test_supersede_archives_old_memory(config: AppConfig) -> None:
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        old = store.create(
            Memory(content="old deadline", agent="agent_a", scope=MemoryScope.SHARED)
        )
        new = store.supersede(
            old.id,
            Memory(content="new deadline Friday", agent="agent_a", scope=MemoryScope.SHARED),
        )
        archived = store.get(old.id)
        assert archived is not None
        assert archived.archived is True
        assert new.supersedes == old.id


def test_timestamp_stored_as_utc(config: AppConfig) -> None:
    init_db(config)
    ts = utc_now()
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        created = store.create(
            Memory(content="timed", agent="a", timestamp=ts, scope=MemoryScope.SHARED)
        )
    assert created.timestamp.tzinfo is not None


def test_delete_memory(config: AppConfig) -> None:
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        mem = store.create(Memory(content="x", agent="a", scope=MemoryScope.SHARED))
        assert store.delete(mem.id) is True
        assert store.get(mem.id) is None


def test_update_memory(config: AppConfig) -> None:
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        mem = store.create(Memory(content="v1", agent="a", scope=MemoryScope.SHARED))
        updated = store.update(mem.model_copy(update={"content": "v2", "importance": 0.9}))
    assert updated.content == "v2"
    assert updated.importance == 0.9


def test_list_by_task(config: AppConfig) -> None:
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        store.create(Memory(content="a", agent="a1", scope=MemoryScope.SHARED, task_id="t1"))
        store.create(Memory(content="b", agent="a2", scope=MemoryScope.LOCAL, task_id="t2"))
        rows = store.list_by_task("t1")
    assert len(rows) == 1
