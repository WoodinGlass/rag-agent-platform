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
    llm_provider: Literal["openai", "anthropic", "groq", "fake"] = "fake"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    # Groq (OpenAI-compatible, has a free tier)
    groq_api_key: str = ""
    # Optional override of the API base URL. When empty:
    #   openai   -> official OpenAI
    #   groq     -> https://api.groq.com/openai/v1
    #   anthropic-> official Anthropic
    llm_base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: int = 30
    llm_max_retries: int = 3

    # --- Agent backend (M6.1) ---
    # state_machine: the deterministic loop in app/agents/agent.py (default)
    # langgraph:     StateGraph-backed loop (requires [langgraph] extra)
    agent_backend: Literal["state_machine", "langgraph"] = "state_machine"

    # --- Vector store ---
    vector_backend: Literal["memory", "chroma", "qdrant"] = "memory"
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "rag_docs"
    # Qdrant (used when VECTOR_BACKEND=qdrant; requires [qdrant] extra)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_docs"
    # Must match the embedder dim (e.g. 1536 for text-embedding-3-small,
    # 128 for FakeEmbedder default).
    qdrant_dim: int = 1536
    qdrant_api_key: str = ""

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

    # --- Tracing (M5.4, M6.3) ---
    # Off by default: no deps needed, no runtime cost.
    otel_enabled: bool = False
    otel_service_name: str = "rag-agent-platform"
    # Exporter: "console" (dev, spans to stdout) | "otlp" (send to collector)
    otel_exporter: Literal["console", "otlp"] = "console"
    # OTLP HTTP endpoint, e.g. http://otel-collector:4318/v1/traces
    otel_otlp_endpoint: str = "http://localhost:4318/v1/traces"

    # --- Streaming ingestion (M6.4) ---
    # Backend: "memory" (offline, tests) | "kafka" (opt-in via [kafka])
    streaming_backend: Literal["memory", "kafka"] = "memory"
    # How long a single poll waits for a new event before returning None.
    streaming_poll_timeout_s: float = 1.0
    # Retries per event before giving up (event is NOT committed on failure).
    streaming_max_retries: int = 3
    # LRU size for dedup. Events seen within this window are skipped.
    streaming_dedup_window: int = 1024
    # Idle sleep between empty polls (seconds).
    streaming_idle_sleep_s: float = 0.05
    # Prefetch buffer size. 0 disables prefetching.
    streaming_prefetch_n: int = 0
    # Total prefetch budget across all partitions (0 = no total cap;
    # per-partition still bound by streaming_prefetch_n).
    streaming_prefetch_budget_total: int = 0

    # Rebalance grace period (seconds): how long to wait for in-flight
    # work before forcing a partition revoke. Only used when the
    # underlying broker drives rebalance (Kafka consumer groups).
    streaming_rebalance_grace_s: float = 5.0

    # Dedup store: memory (LRU, lost on restart) | sqlite (durable)
    streaming_dedup_backend: Literal["memory", "sqlite"] = "memory"
    # Path used when streaming_dedup_backend=sqlite
    streaming_dedup_sqlite_path: str = "./data/dedup.sqlite"
    # TTL for durable dedup entries (seconds). Only used by sqlite.
    streaming_dedup_ttl_s: float = 86400.0

    # --- Request limits (M9.4) ---
    # Off by default: portfolio runs with zero config.
    rate_limit_enabled: bool = False
    # Token bucket per key (tenant id or client IP).
    rate_limit_rps: float = 10.0          # sustained refill rate
    rate_limit_burst: int = 20            # bucket capacity
    # Max request body size in bytes (0 = no limit). Applied to all routes.
    max_body_size_bytes: int = 0

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
