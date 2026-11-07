"""Embedding wrapper for collaborative memory retrieval."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OFFLINE_DIM = 384


class MemoryEmbedder:
    """
    Lazy-loaded sentence-transformers embedder.

    Uses normalized vectors for cosine similarity via inner product (FAISS).
    For BGE models, query prompts are applied when supported by the library.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        offline: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self._offline = (
            offline
            if offline is not None
            else os.getenv("MEMORY_OFFLINE_RETRIEVAL", "").lower() in ("1", "true", "yes")
        )
        self._model = None

    def _is_offline(self) -> bool:
        if self._offline:
            return True
        return os.getenv("MEMORY_OFFLINE_RETRIEVAL", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _load_model(self):  # noqa: ANN202
        if self._model is not None or self._is_offline():
            return
        try:
            from retrieval.model_cache import get_sentence_transformer

            self._model = get_sentence_transformer(self.model_name)
        except Exception as exc:
            logger.warning("Embedding model unavailable (%s); using offline mode", exc)
            self._offline = True

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a search query (BGE query prompt when available)."""
        self._load_model()
        if self._is_offline():
            return self._offline_encode(text)
        assert self._model is not None
        try:
            vector = self._model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                prompt_name="query",
            )[0]
        except TypeError:
            vector = self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return np.asarray(vector, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode document / memory contents."""
        if not texts:
            return np.zeros((0, OFFLINE_DIM), dtype=np.float32)
        self._load_model()
        if self._is_offline():
            return np.stack([self._offline_encode(t) for t in texts], axis=0)
        assert self._model is not None
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def _offline_encode(text: str, dim: int = OFFLINE_DIM) -> np.ndarray:
        """Deterministic bag-of-words style vectors for tests and CI."""
        vec = np.zeros(dim, dtype=np.float32)
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    @property
    def dimension(self) -> int:
        self._load_model()
        if self._is_offline():
            return OFFLINE_DIM
        assert self._model is not None
        return int(self._model.get_sentence_embedding_dimension())
