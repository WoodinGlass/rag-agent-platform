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

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
