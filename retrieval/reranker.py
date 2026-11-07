"""Cross-encoder reranker for top retrieval candidates."""

from __future__ import annotations

import os
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lazy-loaded MS MARCO MiniLM cross-encoder reranker."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        enabled: bool = True,
        offline: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
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
        if self._model is not None or self._is_offline() or not self.enabled:
            return
        try:
            from retrieval.model_cache import get_cross_encoder

            self._model = get_cross_encoder(self.model_name)
        except Exception as exc:
            logger.warning("Reranker unavailable (%s); skipping rerank stage", exc)
            self._offline = True

    def predict_scores(
        self,
        query: str,
        hits: list[dict[str, Any]],
    ) -> list[float] | None:
        """
        Return raw cross-encoder logits aligned with ``hits`` (no reordering).

        Returns ``None`` when the reranker is disabled, offline, or the model
        could not be loaded — callers should treat this as "skip rerank".
        """
        if not hits or not self.enabled or self._is_offline():
            return None
        self._load_model()
        if self._model is None:
            return None
        pairs = [(query, h["content"]) for h in hits]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Naive cross-encoder rerank: sort hits by raw rerank score.

        Kept for backward compatibility and for the ``with_naive_rerank``
        ablation preset. Production retrieval uses
        ``retrieval.rerank_fusion.fuse_rerank`` instead.
        """
        scores = self.predict_scores(query, hits)
        if scores is None:
            return hits[:top_k] if top_k else hits
        reranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for hit, score in reranked:
            entry = dict(hit)
            entry["rerank_score"] = round(float(score), 6)
            out.append(entry)
        if top_k:
            out = out[:top_k]
        return out
