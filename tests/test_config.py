"""Configuration loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import AppConfig, load_config


def test_load_default_config() -> None:
    config = load_config()
    assert config.app_name == "multi-agent-memory"
    assert config.retrieval_top_k == 5


def test_config_from_dict() -> None:
    config = AppConfig.from_dict({"retrieval_top_k": 10, "debug": True})
    assert config.retrieval_top_k == 10
    assert config.debug is True


def test_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_RETRIEVAL_TOP_K", "7")
    config = AppConfig()
    assert config.retrieval_top_k == 7


def test_missing_config_file_returns_defaults(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    config = load_config(missing)
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
