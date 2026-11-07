"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memory.schema import Memory, MemoryScope, MemoryType
from retrieval.context import RetrievalContext
from retrieval.retriever import build_retriever
from storage import init_db, session_scope
from storage.repository import MemoryRepository
from utils.config import AppConfig, load_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str = "0.1.0"


class MemoryQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    task_id: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    include_archived: bool = False


class MemoryQueryResponse(BaseModel):
    query: str
    hits: list[dict[str, Any]] = Field(default_factory=list)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)


class MemoryWriteRequest(BaseModel):
    content: str = Field(..., min_length=1)
    agent: str = Field(..., min_length=1)
    scope: str = "shared"
    task_id: str | None = None
    memory_type: str = "fact"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    parent_memory: str | None = None
    supersedes: str | None = None


class MemoryWriteResponse(BaseModel):
    memory_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: AppConfig = app.state.config
    setup_logging(level=config.log_level)  # type: ignore[arg-type]
    logger.info("Starting %s", config.app_name)
    init_db(config)
    yield
    logger.info("Shutting down %s", config.app_name)


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    application = FastAPI(
        title=cfg.app_name,
        version="0.1.0",
        debug=cfg.debug,
        lifespan=lifespan,
    )
    application.state.config = cfg

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", app_name=cfg.app_name)

    @application.post("/memory/query", response_model=MemoryQueryResponse)
    async def query_memory(body: MemoryQueryRequest) -> MemoryQueryResponse:
        top_k = body.top_k or cfg.retrieval_top_k
        ctx = RetrievalContext(
            agent_id=body.agent_id,
            task_id=body.task_id,
            min_trust=cfg.default_min_trust,
            include_archived=body.include_archived,
        )
        with session_scope(cfg) as session:
            retriever = build_retriever(MemoryRepository(session), cfg)
            result = retriever.retrieve(body.query, ctx, top_k=top_k)
        return MemoryQueryResponse(
            query=body.query,
            hits=result.hits,
            stage_timings_ms={t.name: t.elapsed_ms for t in result.timings},
        )

    @application.post("/memory/write", response_model=MemoryWriteResponse)
    async def write_memory(body: MemoryWriteRequest) -> MemoryWriteResponse:
        try:
            scope = MemoryScope(body.scope)
            memory_type = MemoryType(body.memory_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        from uuid import UUID

        memory = Memory(
            content=body.content,
            agent=body.agent,
            scope=scope,
            task_id=body.task_id,
            memory_type=memory_type,
            confidence=body.confidence,
            importance=body.importance,
            parent_memory=UUID(body.parent_memory) if body.parent_memory else None,
            supersedes=UUID(body.supersedes) if body.supersedes else None,
        )
        with session_scope(cfg) as session:
            repo = MemoryRepository(session)
            repo.add(memory)
            retriever = build_retriever(repo, cfg)
            retriever.index_memory(memory)
            retriever.save_index()
        return MemoryWriteResponse(memory_id=str(memory.id))

    return application


def __getattr__(name: str) -> FastAPI:
    """Lazy default app for ``uvicorn api.main:app`` without import-time side effects."""
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
