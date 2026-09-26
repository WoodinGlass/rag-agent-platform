"""Application configuration via pydantic-settings.

Every knob comes from env / .env. No hardcoded strings in feature code.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["openai", "anthropic", "fake"] = "fake"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: int = 30
    llm_max_retries: int = 3

    # --- Vector store ---
    vector_backend: Literal["memory", "chroma", "qdrant"] = "memory"
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "rag_docs"

    # --- Embedder ---
    embedder_provider: Literal["fake", "openai"] = "fake"
    embedder_dim: int = 128

    # --- Chunking ---
    chunk_size: int = 500
    chunker_version: str = "v1"

    # --- Retrieval ---
    top_k: int = 5

    # --- Reranker (M5.2) ---
    reranker_backend: Literal["identity", "fake", "cross-encoder"] = "identity"
    retrieve_multiplier: int = 2

    # --- Multi-tenant / auth (M5.3) ---
    # Default OFF: the portfolio runs with a single "public" tenant.
    tenant_auth_enabled: bool = False
    # Format: "key1:tenantA,key2:tenantB" — only used when auth is enabled.
    tenant_keys: str = ""
    default_tenant: str = "public"

    # --- Tracing (M5.4) ---
    # Off by default: no deps needed, no runtime cost.
    otel_enabled: bool = False
    otel_service_name: str = "rag-agent-platform"
    # Console exporter prints spans to stdout. Fine for dev; use an OTLP
    # exporter in production (roadmap).
    otel_console_exporter: bool = True

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
