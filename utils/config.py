"""Configuration loader for YAML/JSON settings and environment overrides."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
ENV_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local")


class AppConfig(BaseSettings):
    """
    Application settings.

    Precedence (low → high): YAML config file → ``.env`` → ``.env.local`` →
    process environment. Keys prefixed with ``MEMORY_`` map to fields
    (e.g. ``MEMORY_USE_RERANKER`` → ``use_reranker``).
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "multi-agent-memory"
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "sqlite:///./data/memory.db"
    faiss_index_path: str = "./data/faiss.index"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    offline_retrieval: bool = Field(
        default=False,
        description="Deterministic hash embeddings instead of sentence-transformers.",
    )
    retrieval_top_k: int = Field(default=5, ge=1)
    retrieval_candidate_k: int = Field(default=50, ge=1)
    semantic_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    trust_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    importance_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    recency_half_life_hours: float = Field(default=168.0, gt=0.0)
    use_reranker: bool = False
    default_min_trust: float = Field(default=0.0, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Trust-aware reranker fusion (see retrieval/rerank_fusion.py)
    # ------------------------------------------------------------------
    fusion_semantic_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    fusion_trust_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    reranker_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    fusion_importance_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    preserve_top_trusted_k: int = Field(default=1, ge=0)
    trust_bucket_reranking: bool = False
    trust_bucket_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    trust_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    experiments_runs_dir: str = "./experiments/runs"

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    google_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")

    @field_validator(
        "openai_api_key",
        "groq_api_key",
        "openrouter_api_key",
        "google_api_key",
        "hf_token",
        mode="before",
    )
    @classmethod
    def _empty_secret_to_none(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        return cls(**data)

    def apply_to_os_environ(self) -> None:
        """Mirror retrieval flags into os.environ for legacy readers."""
        # Do not clobber pytest monkeypatch / conftest overrides mid-run.
        if os.getenv("PYTEST_CURRENT_TEST"):
            return
        os.environ["MEMORY_OFFLINE_RETRIEVAL"] = "1" if self.offline_retrieval else "0"
        if self.hf_token is not None:
            token = self.hf_token.get_secret_value()
            if token:
                os.environ.setdefault("HF_TOKEN", token)

    @property
    def retrieval_mode(self) -> str:
        return "offline" if self.offline_retrieval else "semantic"

    def public_summary(self) -> dict[str, Any]:
        """Config snapshot safe for manifests (no secret values)."""
        return {
            "app_name": self.app_name,
            "retrieval_mode": self.retrieval_mode,
            "offline_retrieval": self.offline_retrieval,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "use_reranker": self.use_reranker,
            "fusion_weights": {
                "semantic": self.fusion_semantic_weight,
                "trust": self.fusion_trust_weight,
                "reranker": self.reranker_weight,
                "importance": self.fusion_importance_weight,
            },
            "preserve_top_trusted_k": self.preserve_top_trusted_k,
            "trust_bucket_reranking": self.trust_bucket_reranking,
            "trust_bucket_threshold": self.trust_bucket_threshold,
            "trust_threshold": self.trust_threshold,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_candidate_k": self.retrieval_candidate_k,
            "experiments_runs_dir": self.experiments_runs_dir,
            "has_openai_key": bool(
                self.openai_api_key and self.openai_api_key.get_secret_value()
            ),
        }


def _bootstrap_dotenv() -> None:
    """Load ``.env`` then ``.env.local`` (local overrides)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = PROJECT_ROOT / ".env"
    local_path = PROJECT_ROOT / ".env.local"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    # Never load .env.local during pytest — it may enable real models and break CI.
    if local_path.is_file() and not os.getenv("PYTEST_CURRENT_TEST"):
        load_dotenv(local_path, override=True)


def _resolve_config_path(config_path: Path | str | None) -> Path:
    if config_path is not None:
        return Path(config_path)
    profile = os.getenv("MEMORY_EXPERIMENT_CONFIG", "").strip()
    if profile == "experiment" and EXPERIMENT_CONFIG_PATH.is_file():
        return EXPERIMENT_CONFIG_PATH
    return DEFAULT_CONFIG_PATH


def _load_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        result = yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
    if path.suffix == ".json":
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _coerce_bool_env(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


@lru_cache
def load_config(config_path: Path | str | None = None) -> AppConfig:
    """
    Load configuration from YAML, dotenv files, and environment.

    Call ``reload_config()`` after editing ``.env.local`` in a long-lived process.
    """
    _bootstrap_dotenv()
    path = _resolve_config_path(config_path)
    file_data = _load_file(path) if path.exists() else {}

    # Build from YAML, then let explicit env / dotenv override (YAML must not win over .env.local).
    config = AppConfig(**file_data)

    env_overrides: dict[str, Any] = {}
    for env_name, field in (
        ("MEMORY_OFFLINE_RETRIEVAL", "offline_retrieval"),
        ("MEMORY_USE_RERANKER", "use_reranker"),
    ):
        value = _coerce_bool_env(env_name)
        if value is not None:
            env_overrides[field] = value
    # Default tests to offline embeddings unless MEMORY_OFFLINE_RETRIEVAL is set explicitly.
    if os.getenv("PYTEST_CURRENT_TEST") and "offline_retrieval" not in env_overrides:
        env_overrides["offline_retrieval"] = True
    if env_overrides:
        config = config.model_copy(update=env_overrides)

    config.apply_to_os_environ()
    return config


def reload_config(config_path: Path | str | None = None) -> AppConfig:
    """Clear cache and reload settings."""
    load_config.cache_clear()
    return load_config(config_path)
