"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from utils.config import AppConfig, load_config


@pytest.fixture(autouse=True)
def _offline_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid downloading transformer models in CI."""
    monkeypatch.setenv("MEMORY_OFFLINE_RETRIEVAL", "1")


@pytest.fixture(autouse=True)
def _clear_config_cache() -> Generator[None, None, None]:
    load_config.cache_clear()
    yield
    load_config.cache_clear()


@pytest.fixture(autouse=True)
def _reset_db_engine() -> Generator[None, None, None]:
    import storage.database as db

    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._SessionLocal = None
    yield
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._SessionLocal = None


@pytest.fixture
def test_config() -> AppConfig:
    return AppConfig(
        app_name="multi-agent-memory-test",
        debug=True,
        log_level="WARNING",
        database_url="sqlite:///:memory:",
        faiss_index_path="./data/test.faiss",
        offline_retrieval=True,
        use_reranker=False,
    )


@pytest.fixture(autouse=True)
def _init_tables(test_config: AppConfig) -> Generator[None, None, None]:
    from storage.init_db import init_db

    init_db(test_config)
    yield


@pytest.fixture
def client(test_config: AppConfig) -> Generator[TestClient, None, None]:
    app = create_app(config=test_config)
    with TestClient(app) as test_client:
        yield test_client
