"""Real Mem0 OSS backend (mem0ai SDK)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from baselines.runtime import configure_torch_safety
from memory.schema import Memory
from retrieval.context import RetrievalContext
from retrieval.retriever import RetrievalResult, RetrievalStageTiming
from storage.repository import MemoryRepository
from utils.config import AppConfig
from utils.logging import get_logger

logger = get_logger(__name__)

# Small local model for benchmarks when real BGE is unavailable.
MEM0_BENCH_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MEM0_EMBED_DIM = 384

# One Mem0 client per cache directory per process (avoids 12× torch reloads per sweep).
_MEM0_CLIENT_CACHE: dict[str, Any] = {}


def _mem0_available() -> bool:
    try:
        import mem0  # noqa: F401

        return True
    except ImportError:
        return False


class Mem0Backend:
    """
    Real Mem0 memory layer via ``mem0ai``.

    - Writes are mirrored from the benchmark SQL store with ``infer=False`` (raw facts).
    - Each agent is a Mem0 ``user_id`` (user-scoped memory).
    - Search uses Mem0 vector search + metadata filters (``task_id``, etc.).
    """

    technique = "mem0"

    def __init__(
        self,
        repository: MemoryRepository,
        config: AppConfig,
        *,
        work_dir: Path,
        mem0_cache_dir: Path | str | None = None,
    ) -> None:
        if not _mem0_available():
            raise ImportError(
                "mem0ai is required for the mem0 technique. "
                "Install with: pip install -e '.[baselines]'"
            )

        configure_torch_safety()

        self._repo = repository
        self._config = config
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Dedicated small embedder — do not load BGE again (collaborative already uses it).
        embed_model = MEM0_BENCH_EMBED_MODEL
        embed_dims = MEM0_EMBED_DIM

        cache_root = Path(mem0_cache_dir or self._work_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MEM0_DIR", str(cache_root / "mem0_home"))

        cache_key = str(cache_root.resolve())
        if cache_key not in _MEM0_CLIENT_CACHE:
            from mem0 import Memory

            mem0_config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": str(cache_root / "qdrant"),
                        "on_disk": True,
                        "embedding_model_dims": embed_dims,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": embed_model,
                        "embedding_dims": embed_dims,
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini"),
                        "api_key": os.getenv("OPENAI_API_KEY")
                        or os.getenv("MEM0_OPENAI_API_KEY")
                        or "mem0-benchmark-no-key",
                    },
                },
                "history_db_path": str(cache_root / "mem0_history.db"),
            }
            logger.info(
                "Initializing Mem0 client (one per benchmark run): embed=%s cache=%s",
                embed_model,
                cache_key,
            )
            _MEM0_CLIENT_CACHE[cache_key] = Memory.from_config(mem0_config)

        self._memory = _MEM0_CLIENT_CACHE[cache_key]
        self._synced_ids: set[str] = set()

    def index_memory(self, memory: Memory) -> None:
        if memory.archived:
            self._synced_ids.discard(str(memory.id))
            return
        self._add_to_mem0(memory)

    def _add_to_mem0(self, memory: Memory) -> None:
        metadata = {
            "memory_id": str(memory.id),
            "task_id": memory.task_id or "",
            "scope": memory.scope.value,
            "agent": memory.agent,
            "confidence": memory.confidence,
            "importance": memory.importance,
            "archived": memory.archived,
        }
        self._memory.add(
            memory.content,
            user_id=memory.agent,
            infer=False,
            metadata=metadata,
        )
        self._synced_ids.add(str(memory.id))

    def sync_from_repository(self, *, task_id: str | None = None) -> int:
        if not task_id:
            return 0
        records = self._repo.list_by_task(task_id, include_archived=True)

        count = 0
        for memory in records:
            mem_id = str(memory.id)
            if memory.archived:
                self._synced_ids.discard(mem_id)
                continue
            if mem_id not in self._synced_ids:
                self._add_to_mem0(memory)
                count += 1
        return count

    def retrieve(
        self,
        query: str,
        ctx: RetrievalContext,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        k = top_k or self._config.retrieval_top_k
        result = RetrievalResult(query=query)

        start = time.perf_counter()
        if ctx.task_id:
            self.sync_from_repository(task_id=ctx.task_id)
        result.timings.append(
            RetrievalStageTiming("candidate_pool", (time.perf_counter() - start) * 1000)
        )

        filters: dict[str, Any] = {"user_id": ctx.agent_id}
        if ctx.task_id:
            filters["task_id"] = ctx.task_id

        start = time.perf_counter()
        try:
            raw = self._memory.search(query, filters=filters, limit=k * 3)
        except TypeError:
            # Older mem0 API
            raw = self._memory.search(query, user_id=ctx.agent_id, limit=k * 3)

        entries = raw.get("results", raw) if isinstance(raw, dict) else raw
        result.timings.append(
            RetrievalStageTiming("hybrid_search", (time.perf_counter() - start) * 1000)
        )

        hits: list[dict[str, Any]] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            meta = entry.get("metadata") or {}
            if meta.get("archived"):
                continue
            if ctx.task_id and meta.get("task_id") not in (ctx.task_id, ""):
                continue
            mem_id = meta.get("memory_id") or entry.get("id", "")
            hits.append(
                {
                    "memory_id": str(mem_id),
                    "content": entry.get("memory", entry.get("content", "")),
                    "agent": meta.get("agent", ctx.agent_id),
                    "scope": meta.get("scope", "shared"),
                    "score": float(entry.get("score", 0.0)),
                    "confidence": float(meta.get("confidence", 1.0)),
                    "importance": float(meta.get("importance", 0.5)),
                }
            )
            if len(hits) >= k:
                break

        result.hits = hits
        return result

    def close(self) -> None:
        pass

    def backend_info(self) -> dict[str, Any]:
        return {
            "implementation": "mem0ai_sdk",
            "infer": False,
            "work_dir": str(self._work_dir),
            "embed_model": MEM0_BENCH_EMBED_MODEL,
        }
