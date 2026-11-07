"""Shared scaffolding for collaborative-memory benchmark scenarios."""

from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from evaluation.ablation_metrics import attach_query_trace
from evaluation.harness import BenchmarkReport
from evaluation.metrics import CollaborativeMetrics
from baselines.factory import create_backend
from baselines.protocol import MemoryBackend
from baselines.runtime import configure_torch_safety
from memory.contradictions import ContradictionRegistry
from retrieval.ablation import RetrievalAblation
from retrieval.context import RetrievalContext
from retrieval.retriever import RetrievalResult
from storage.repository import MemoryRepository
from utils.config import AppConfig


def make_bench_config(scenario_name: str, *, faiss_dir: Path) -> AppConfig:
    """
    Bench config: inherits ``.env.local`` retrieval settings, isolates storage.

    Uses in-memory SQLite and a per-scenario FAISS path under a temp directory.
    """
    from utils.config import load_config

    base = load_config()
    # Tests force offline via conftest; benchmarks use .env.local otherwise.
    offline = os.getenv("MEMORY_OFFLINE_RETRIEVAL", "").lower() in ("1", "true", "yes")
    if os.getenv("PYTEST_CURRENT_TEST"):
        offline = True
    return base.model_copy(
        update={
            "app_name": f"bench_{scenario_name}",
            "log_level": "WARNING",
            "database_url": "sqlite:///:memory:",
            "faiss_index_path": str(faiss_dir / "faiss.index"),
            "offline_retrieval": offline,
        }
    )


def reset_storage_engine() -> None:
    """Drop the cached SQLAlchemy engine so each scenario starts clean."""
    from storage import database as db

    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._SessionLocal = None


class BaseScenario(ABC):
    """
    Minimal base class providing isolated DB + FAISS, deterministic config.

    Subclasses implement ``run()`` and may declare:
      - ``name``      : scenario identifier
      - ``expected``  : dict of metric_name -> threshold for pass/fail
    """

    name: str = "base_scenario"
    expected: dict[str, float] = {}

    def __init__(
        self,
        *,
        technique: str | None = None,
        ablation: RetrievalAblation | None = None,
        contradiction_registry: ContradictionRegistry | None = None,
        mem0_cache_dir: str | Path | None = None,
    ) -> None:
        self._tmp_dir: str | None = None
        self._config: AppConfig | None = None
        self.technique = technique
        self.ablation = ablation
        self._contradiction_registry = contradiction_registry
        self._mem0_cache_dir = Path(mem0_cache_dir) if mem0_cache_dir else None
        self._backend: MemoryBackend | None = None

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            raise RuntimeError(f"{self.name}: setup() has not been called")
        return self._config

    def setup(self) -> None:
        configure_torch_safety()
        reset_storage_engine()
        self._tmp_dir = tempfile.mkdtemp(prefix=f"bench_{self.name}_")
        self._config = make_bench_config(self.name, faiss_dir=Path(self._tmp_dir))
        from storage import init_db

        init_db(self._config)

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
        reset_storage_engine()

    def build_backend(
        self,
        session: Any,
        *,
        registry: ContradictionRegistry | None = None,
    ) -> MemoryBackend:
        """Build a real memory backend (technique or ablation-driven collaborative)."""
        reg = registry or self._contradiction_registry
        repo = MemoryRepository(session)
        work_dir = Path(self._tmp_dir or ".")
        if self.technique:
            self._backend = create_backend(
                self.technique,
                repo,
                self.config,
                work_dir=work_dir,
                contradiction_registry=reg,
                ablation=self.ablation,
                mem0_cache_dir=self._mem0_cache_dir,
            )
        else:
            self._backend = create_backend(
                "collaborative",
                repo,
                self.config,
                work_dir=work_dir,
                contradiction_registry=reg,
                ablation=self.ablation,
                mem0_cache_dir=self._mem0_cache_dir,
            )
        return self._backend

    def build_retriever(
        self,
        session: Any,
        *,
        registry: ContradictionRegistry | None = None,
    ) -> MemoryBackend:
        """Backward-compatible alias: returns a ``MemoryBackend`` with ``retrieve()``."""
        return self.build_backend(session, registry=registry)

    @staticmethod
    def retrieve_traced(
        backend: MemoryBackend,
        query: str,
        ctx: RetrievalContext,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, dict[str, float]]:
        """Run retrieval and return stage latency map."""
        result = backend.retrieve(query, ctx, top_k=top_k)
        from evaluation.ablation_metrics import stage_timings_from_result

        return result, stage_timings_from_result(result)

    @staticmethod
    def enrich_trace(trace: dict[str, Any], result: RetrievalResult) -> dict[str, Any]:
        return attach_query_trace(trace, result)

    @abstractmethod
    def run(self) -> BenchmarkReport | CollaborativeMetrics: ...


# ---------------------------------------------------------------------------
# Deterministic time helpers
# ---------------------------------------------------------------------------

BENCH_EPOCH = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


def at(minutes: int = 0, hours: int = 0, days: int = 0) -> datetime:
    """Deterministic timestamp offset from BENCH_EPOCH."""
    return BENCH_EPOCH + timedelta(minutes=minutes, hours=hours, days=days)


def serialize_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip volatile fields (none currently) and round numeric scores."""
    out: list[dict[str, Any]] = []
    for hit in hits:
        clean = dict(hit)
        for key, value in list(clean.items()):
            if isinstance(value, float):
                clean[key] = round(value, 6)
        out.append(clean)
    return out
