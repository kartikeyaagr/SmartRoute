import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # Classifier
    classifier: str = "keyword"  # "keyword" or "deberta"
    classifier_fallback_timeout_ms: float = 2000.0  # fallback if DeBERTa warmup > this

    # Cascade
    cascade_verifier: bool = True  # set False to disable verifier (routing-only mode)
    verifier_confidence_threshold: int = 4  # escalate if verifier score < this

    # Benchmark
    benchmark_concurrency: int = 5   # SMARTROUTE_BENCHMARK_CONCURRENCY
    benchmark_cost_ceiling_usd: float = 20.0

    # Provider timeouts
    model_timeout_s: float = 30.0
    verifier_timeout_s: float = 30.0

    # Cache (v0.2)
    cache_backend: str = "none"  # "none" | "memory" | "pgvector"
    cache_similarity_threshold: float = 0.95  # reserved for pgvector semantic cache
    cache_ttl_seconds: float = 3600.0  # 1 hour default
    cache_max_size: int = 1000  # max entries for in-memory LRU cache

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    # When set, all /v1/* requests must carry `Authorization: Bearer <key>`.
    # Leave empty to run unauthenticated (local dev only).
    server_api_key: str = ""


settings = Settings()

# Propagate .env values to os.environ so LiteLLM can pick them up.
# pydantic_settings reads .env into model fields but does not mutate os.environ.
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
if settings.gemini_api_key:
    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
if settings.groq_api_key:
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
