"""Construct real memory backends for benchmarks and demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from baselines.a_mem_backend import AMemLinkedBackend
from baselines.collaborative_backend import CollaborativeBackend
from baselines.hybrid_bm25_backend import HybridBM25Backend
from baselines.isolated_memory_backend import IsolatedMemoryBackend
from baselines.mem0_backend import Mem0Backend, _mem0_available
from baselines.memgpt_tiered_backend import MemGPTTieredBackend
from baselines.protocol import MemoryBackend
from baselines.simple_rag_backend import SimpleRAGBackend
from baselines.techniques import list_techniques, variant_for_technique
from memory.contradictions import ContradictionRegistry
from retrieval.ablation import RetrievalAblation, get_variant
from storage.repository import MemoryRepository
from utils.config import AppConfig


def create_backend(
    technique: str,
    repository: MemoryRepository,
    config: AppConfig,
    *,
    work_dir: Path | str,
    contradiction_registry: ContradictionRegistry | None = None,
    ablation: RetrievalAblation | None = None,
    mem0_cache_dir: Path | str | None = None,
) -> MemoryBackend:
    """
    Instantiate a real backend for ``technique``.

    - ``collaborative``: full pipeline (optional ``ablation`` override).
    - ``simple_rag``: standalone FAISS semantic RAG.
    - ``mem0``: Mem0 OSS SDK (``mem0ai`` package).
    - ``hybrid_bm25``: BM25 + dense RRF fusion.
    - ``memgpt_tiered``: pinned core + archival vector search.
    - ``isolated_memory``: author-only retrieval (no shared tier).
    - ``a_mem_linked``: semantic seed + tag-linked neighbor expansion.
    """
    key = technique.strip().lower()
    if key not in list_techniques():
        raise KeyError(f"Unknown technique {technique!r}. Choose from: {list_techniques()}")

    work = Path(work_dir)

    if key == "collaborative":
        preset = ablation
        if preset is None:
            variant = variant_for_technique(key)
            preset = get_variant(variant)
        return CollaborativeBackend(
            repository,
            config,
            ablation=preset,
            contradiction_registry=contradiction_registry,
        )

    if key == "simple_rag":
        return SimpleRAGBackend(
            repository,
            config,
            faiss_path=str(work / "simple_rag.faiss"),
        )

    if key == "mem0":
        if not _mem0_available():
            raise ImportError(
                "Install Mem0 for real mem0 benchmarks: pip install -e '.[baselines]'"
            )
        return Mem0Backend(
            repository,
            config,
            work_dir=work / "mem0",
            mem0_cache_dir=mem0_cache_dir,
        )

    if key == "hybrid_bm25":
        return HybridBM25Backend(
            repository,
            config,
            faiss_path=str(work / "hybrid_bm25.faiss"),
        )

    if key == "memgpt_tiered":
        return MemGPTTieredBackend(
            repository,
            config,
            faiss_path=str(work / "memgpt_tiered.faiss"),
        )

    if key == "isolated_memory":
        return IsolatedMemoryBackend(
            repository,
            config,
            faiss_path=str(work / "isolated_memory.faiss"),
        )

    if key == "a_mem_linked":
        return AMemLinkedBackend(
            repository,
            config,
            faiss_path=str(work / "a_mem_linked.faiss"),
        )

    raise KeyError(f"Unhandled technique: {technique!r}")


def backend_for_technique_label(technique: str) -> dict[str, Any]:
    """Static description for reports."""
    descriptions = {
        "collaborative": "SQLite + FAISS + trust-aware fusion + lineage/contradiction filters",
        "simple_rag": "SQLite + FAISS semantic search only (no trust/lineage/rerank)",
        "mem0": "mem0ai SDK + Qdrant + HuggingFace embeddings (infer=False raw adds)",
        "hybrid_bm25": "BM25 sparse + FAISS dense fused with RRF (no trust/lineage)",
        "memgpt_tiered": "Pinned recent core slots + archival FAISS search (Letta/MemGPT-lite)",
        "isolated_memory": "FAISS per-agent only; shared memories invisible (Collaborative Memory 2025)",
        "a_mem_linked": "FAISS seed + tag-overlap graph expansion (A-Mem Zettelkasten-lite)",
    }
    return {
        "technique": technique,
        "description": descriptions.get(technique, ""),
        "real_implementation": True,
    }
