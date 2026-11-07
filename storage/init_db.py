"""Database initialization for the collaborative memory store."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from storage.database import get_engine
from storage.models import Base
from utils.config import AppConfig, load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return
    raw = database_url.removeprefix("sqlite:///")
    if raw == ":memory:":
        return
    Path(raw).parent.mkdir(parents=True, exist_ok=True)


def init_db(config: AppConfig | None = None, *, engine: Engine | None = None) -> Engine:
    """
    Create database tables if they do not exist.

    Args:
        config: Application config (loads default when omitted).
        engine: Optional pre-built engine; otherwise resolved from config.

    Returns:
        The SQLAlchemy engine bound to the initialized database.
    """
    cfg = config or load_config()
    _ensure_sqlite_parent_dir(cfg.database_url)

    db_engine = engine or get_engine(cfg)
    Base.metadata.create_all(bind=db_engine)
    _add_missing_columns(db_engine)
    logger.info("Initialized memory database at %s", cfg.database_url)
    return db_engine


def _add_missing_columns(engine: Engine) -> None:
    """Add nullable columns introduced after a database was created (create_all won't).

    ponytail: additive ALTERs only; switch to Alembic if a column ever needs changing.
    """
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        present = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in present and col.nullable:
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}"
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                logger.info("Added column %s.%s", table.name, col.name)


def reset_db(config: AppConfig | None = None, *, engine: Engine | None = None) -> Engine:
    """
    Drop and recreate all tables (research / test helper only).

    Warning: destructive — all memories are removed.
    """
    cfg = config or load_config()
    db_engine = engine or get_engine(cfg)
    Base.metadata.drop_all(bind=db_engine)
    Base.metadata.create_all(bind=db_engine)
    logger.warning("Reset memory database at %s", cfg.database_url)
    return db_engine
