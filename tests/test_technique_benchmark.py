"""Tests for memory technique presets and mapping."""

from __future__ import annotations

import pytest

from baselines.techniques import TECHNIQUE_VARIANTS, list_techniques, variant_for_technique
from retrieval.ablation import ABLATION_VARIANTS, get_variant


def test_technique_mapping() -> None:
    assert set(list_techniques()) == {
        "collaborative",
        "simple_rag",
        "mem0",
        "hybrid_bm25",
        "memgpt_tiered",
        "isolated_memory",
        "a_mem_linked",
    }
    assert variant_for_technique("collaborative") == "full"
    assert "full" in ABLATION_VARIANTS
    for preset in ("simple_rag", "mem0"):
        assert preset in ABLATION_VARIANTS
        assert variant_for_technique(preset) == preset


def test_new_presets_exist() -> None:
    simple = get_variant("simple_rag")
    assert simple.name == "simple_rag"
    assert simple.enable_lineage_filter is False
    assert simple.enable_trust_scoring is False

    mem0 = get_variant("mem0")
    assert mem0.name == "mem0"
    assert mem0.agent_scoped_only is True


def test_unknown_technique_raises() -> None:
    with pytest.raises(KeyError, match="Unknown technique"):
        variant_for_technique("unknown")
