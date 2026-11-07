"""Backward-compatible export of the collaborative retrieval pipeline."""

from retrieval.retriever import (
    CollaborativeMemoryRetriever,
    CollaborativeRetrievalPipeline,
    RetrievalResult,
    RetrievalStageTiming,
    build_retriever,
)

__all__ = [
    "CollaborativeMemoryRetriever",
    "CollaborativeRetrievalPipeline",
    "RetrievalResult",
    "RetrievalStageTiming",
    "build_retriever",
]
