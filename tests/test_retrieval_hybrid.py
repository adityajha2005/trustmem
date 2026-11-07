"""Hybrid local + shared fusion tests."""

from __future__ import annotations

from retrieval.hybrid_search import fuse_pools


def test_fuse_pools_takes_max_per_id() -> None:
    fused = fuse_pools([("a", 0.8)], [("a", 0.6), ("b", 0.7)])
    assert fused["a"] == 0.8 * 1.05  # local pool boost wins over shared score
    assert fused["b"] == 0.7
