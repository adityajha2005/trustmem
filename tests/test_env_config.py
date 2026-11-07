"""Tests for .env.local loading and retrieval mode toggles."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import AppConfig, load_config, reload_config


def test_offline_retrieval_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    load_config.cache_clear()
    monkeypatch.setenv("MEMORY_OFFLINE_RETRIEVAL", "1")
    cfg = AppConfig()
    assert cfg.offline_retrieval is True
    assert cfg.retrieval_mode == "offline"


def test_dotenv_local_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    load_config.cache_clear()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        "offline_retrieval: false\nuse_reranker: false\n", encoding="utf-8"
    )
    (tmp_path / ".env").write_text("MEMORY_USE_RERANKER=false\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text(
        "MEMORY_OFFLINE_RETRIEVAL=1\nMEMORY_USE_RERANKER=true\n",
        encoding="utf-8",
    )

    # Point loader at tmp project layout
    import utils.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", tmp_path / "configs" / "default.yaml")
    monkeypatch.setattr(cfg_mod, "ENV_FILES", (tmp_path / ".env", tmp_path / ".env.local"))

    cfg = reload_config()
    assert cfg.offline_retrieval is True
    assert cfg.use_reranker is True


def test_public_summary_excludes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    load_config.cache_clear()
    cfg = AppConfig()
    summary = cfg.public_summary()
    assert "sk-test" not in str(summary)
    assert summary["has_openai_key"] is True


def test_use_reranker_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    load_config.cache_clear()
    monkeypatch.setenv("MEMORY_USE_RERANKER", "true")
    cfg = load_config()
    assert cfg.use_reranker is True


def test_apply_to_os_environ_syncs_offline_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMORY_OFFLINE_RETRIEVAL", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    cfg = AppConfig(offline_retrieval=True)
    cfg.apply_to_os_environ()
    assert __import__("os").environ["MEMORY_OFFLINE_RETRIEVAL"] == "1"
