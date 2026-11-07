"""SQLite engine and session management (WAL-enabled, local-first)."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from utils.config import AppConfig, load_config

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _sqlite_path(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite:///"):
        return None
    raw = database_url.removeprefix("sqlite:///")
    if raw == ":memory:":
        return None
    return Path(raw)


def _enable_wal(engine: Engine) -> None:
    """Enable WAL for file-backed SQLite."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine(config: AppConfig | None = None) -> Engine:
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    cfg = config or load_config()
    db_path = _sqlite_path(cfg.database_url)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        cfg.database_url,
        connect_args={"check_same_thread": False} if cfg.database_url.startswith("sqlite") else {},
    )
    if cfg.database_url.startswith("sqlite"):
        _enable_wal(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=True, autocommit=False)
    return _engine


def get_session_factory(config: AppConfig | None = None) -> sessionmaker[Session]:
    get_engine(config)
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope(config: AppConfig | None = None) -> Generator[Session, None, None]:
    factory = get_session_factory(config)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
