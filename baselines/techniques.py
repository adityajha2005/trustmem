"""Map user-facing technique names to real backend implementations."""

from __future__ import annotations

# Collaborative uses production ``full`` ablation; others use dedicated backends in ``baselines/``.
TECHNIQUE_VARIANTS: dict[str, str] = {
    "collaborative": "full",
    "simple_rag": "simple_rag",
    "mem0": "mem0",
    "hybrid_bm25": "hybrid_bm25",
    "memgpt_tiered": "memgpt_tiered",
    "isolated_memory": "isolated_memory",
    "a_mem_linked": "a_mem_linked",
}

TECHNIQUE_LABELS: dict[str, str] = {
    "collaborative": "Collaborative (trust-aware)",
    "simple_rag": "Simple RAG (FAISS)",
    "mem0": "Mem0 OSS SDK",
    "hybrid_bm25": "Hybrid BM25 + dense (INMS-style)",
    "memgpt_tiered": "MemGPT tiered (core + archival)",
    "isolated_memory": "Isolated memory (no shared tier)",
    "a_mem_linked": "A-Mem linked (tag expansion)",
}


def list_techniques() -> list[str]:
    return list(TECHNIQUE_VARIANTS.keys())


def technique_label(technique: str) -> str:
    return TECHNIQUE_LABELS.get(technique, technique)


def variant_for_technique(technique: str) -> str:
    key = technique.strip().lower()
    if key not in TECHNIQUE_VARIANTS:
        raise KeyError(
            f"Unknown technique {technique!r}. Choose from: {list_techniques()}"
        )
    return TECHNIQUE_VARIANTS[key]
