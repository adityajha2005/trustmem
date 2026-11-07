"""Process-wide HF model cache."""

from __future__ import annotations

import pytest

from retrieval.model_cache import cache_stats, clear_model_cache


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_model_cache()
    yield
    clear_model_cache()


def test_embedder_reuses_cached_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sentence_transformers")
    from retrieval.embedder import MemoryEmbedder
    from retrieval.model_cache import _SENTENCE_TRANSFORMERS

    monkeypatch.delenv("MEMORY_OFFLINE_RETRIEVAL", raising=False)

    a = MemoryEmbedder("sentence-transformers/all-MiniLM-L6-v2", offline=False)
    b = MemoryEmbedder("sentence-transformers/all-MiniLM-L6-v2", offline=False)
    a.encode_query("hello")
    b.encode_query("world")

    assert len(_SENTENCE_TRANSFORMERS) == 1
    assert a._model is b._model
    assert cache_stats()["sentence_transformers"] == 1
