"""Process-wide cache for Hugging Face embedding / reranker models."""

from __future__ import annotations

from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

# Loaded once per (model_name) per process — avoids 70× reload in LLM eval sweeps.
_SENTENCE_TRANSFORMERS: dict[str, Any] = {}
_CROSS_ENCODERS: dict[str, Any] = {}


def get_sentence_transformer(model_name: str) -> Any:
    """Return a shared ``SentenceTransformer`` instance for ``model_name``."""
    if model_name in _SENTENCE_TRANSFORMERS:
        return _SENTENCE_TRANSFORMERS[model_name]
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model (cached for process): %s", model_name)
    model = SentenceTransformer(model_name)
    _SENTENCE_TRANSFORMERS[model_name] = model
    return model


def get_cross_encoder(model_name: str) -> Any:
    """Return a shared ``CrossEncoder`` instance for ``model_name``."""
    if model_name in _CROSS_ENCODERS:
        return _CROSS_ENCODERS[model_name]
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker (cached for process): %s", model_name)
    model = CrossEncoder(model_name)
    _CROSS_ENCODERS[model_name] = model
    return model


def clear_model_cache() -> None:
    """Drop cached models (for tests or memory pressure)."""
    _SENTENCE_TRANSFORMERS.clear()
    _CROSS_ENCODERS.clear()


def cache_stats() -> dict[str, int]:
    return {
        "sentence_transformers": len(_SENTENCE_TRANSFORMERS),
        "cross_encoders": len(_CROSS_ENCODERS),
    }
