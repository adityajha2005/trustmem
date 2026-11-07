"""FAISS-backed vector index for memory retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


class FaissVectorStore:
    """
    Inner-product index over L2-normalized vectors (cosine similarity).

    Maintains a parallel id map: FAISS row index -> memory UUID string.
    """

    def __init__(self, index_path: Path | str, *, dimension: int) -> None:
        self.index_path = Path(index_path)
        self.id_map_path = self.index_path.with_suffix(".ids.json")
        self.dimension = dimension
        self._index: faiss.IndexFlatIP | None = None
        self._ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self._ids)

    def _ensure_index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add(self, memory_id: UUID | str, vector: np.ndarray) -> None:
        """Add or replace a vector for a memory id."""
        mem_id = str(memory_id)
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {vec.shape[1]}")

        if mem_id in self._ids:
            self.remove(mem_id)

        index = self._ensure_index()
        index.add(vec)
        self._ids.append(mem_id)

    def remove(self, memory_id: UUID | str) -> None:
        """
        Remove a memory from the index by rebuilding without it.

        FAISS Flat index lacks efficient deletion; fine for research scale.
        """
        mem_id = str(memory_id)
        if mem_id not in self._ids:
            return
        if self._index is None or self._index.ntotal == 0:
            return

        vectors = self._index.reconstruct_n(0, self._index.ntotal)
        keep_ids: list[str] = []
        keep_vecs: list[np.ndarray] = []
        for idx, mid in enumerate(self._ids):
            if mid != mem_id:
                keep_ids.append(mid)
                keep_vecs.append(vectors[idx])
        self._index = faiss.IndexFlatIP(self.dimension)
        self._ids = []
        if keep_vecs:
            batch = np.stack(keep_vecs, axis=0).astype(np.float32)
            self._index.add(batch)
            self._ids = keep_ids

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 20,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return (memory_id, similarity_score) pairs sorted by score descending."""
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k if allowed_ids is None else max(top_k * 3, top_k), self._index.ntotal)
        scores, indices = self._index.search(q, k)

        hits: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._ids):
                continue
            mem_id = self._ids[idx]
            if allowed_ids is not None and mem_id not in allowed_ids:
                continue
            hits.append((mem_id, float(score)))
            if len(hits) >= top_k:
                break
        return hits

    def save(self) -> None:
        """Persist index and id map to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None and self._index.ntotal > 0:
            faiss.write_index(self._index, str(self.index_path))
        elif self.index_path.exists():
            self.index_path.unlink(missing_ok=True)
        self.id_map_path.write_text(json.dumps(self._ids), encoding="utf-8")
        logger.debug("Saved FAISS index (%d vectors) to %s", len(self._ids), self.index_path)

    def load(self) -> bool:
        """Load index from disk if present."""
        if not self.index_path.exists() or not self.id_map_path.exists():
            return False
        self._index = faiss.read_index(str(self.index_path))
        self._ids = json.loads(self.id_map_path.read_text(encoding="utf-8"))
        logger.info("Loaded FAISS index with %d vectors from %s", len(self._ids), self.index_path)
        return True

    def sync_memories(
        self,
        memory_ids: list[str],
        vectors: np.ndarray,
    ) -> None:
        """Bulk add vectors aligned with memory_ids."""
        for mem_id, vec in zip(memory_ids, vectors, strict=True):
            self.add(mem_id, vec)
