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
    togetherai_api_key: str = ""
    openrouter_api_key: str = ""
    # TypeSafe's own key for Jev (api.typesafe.ai). When set, Jev calls go there instead
    # of through OpenRouter; task-ranked routing still needs OPENROUTER_API_KEY for the
    # benchmarks/models data and for calling the chosen model.
    typesafe_api_key: str = ""
    # Jev input price for cost accounting on TypeSafe calls (its response has no cost).
    jev_price_per_mtok: float = 0.04198

    # Router dispatch: "decision" (two-layer, default) | "cascade" | "taskrank"
    # taskrank = Decisions-API task + effort classification -> OpenRouter task rankings
    # -> effort-based cost-tier cap -> call via OpenRouter. Needs OPENROUTER_API_KEY.
    dispatch: str = "decision"
    decision_model: str = "~typesafe/jev-latest"

    # Jev check of middle-tier answers (dispatch="decision"); escalates to frontier on fail.
    jev_escalation: bool = False
    # Escalate when Jev's P(correct) is below this. Unset = derived from the catalog:
    # 1 - frontier_cost / lambda_wrong_answer_usd. Calibrate it; Jev's probabilities
    # are not calibrated to your traffic.
    jev_escalation_threshold: float | None = None
    # Which checker(s) score middle answers, comma-separated: jev, learned, consistency.
    # Several -> mean P(correct). "both" = "jev,learned". "learned" needs
    # src/smartroute/data/checker.joblib (harness/train_checker.py).
    escalation_checker: str = "jev"
    # Self-consistency: extra samples drawn and compared with the middle answer.
    # sc_source "middle" = true self-consistency; "cheap" = cross-model agreement (cheaper).
    sc_samples: int = 2
    sc_source: str = "middle"
    sc_temperature: float = 0.7

    # Model catalog — path to models.yaml. Empty = repo-root models.yaml.
    catalog_path: str = ""

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
    cache_backend: str = "none"  # "none" | "memory" | "redis" | "pgvector"
    cache_similarity_threshold: float = 0.95  # reserved for pgvector semantic cache
    cache_ttl_seconds: float = 3600.0  # 1 hour default
    cache_max_size: int = 1000  # max entries for in-memory LRU cache
    redis_url: str = "redis://localhost:6379/0"  # CACHE_BACKEND=redis

    # Database — leave empty to run without Postgres (JSONL logging only)
    database_url: str = ""

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    log_level: str = "INFO"
    # When set, all /v1/* requests must carry `Authorization: Bearer <key>`.
    # Leave empty to run unauthenticated (local dev only).
    server_api_key: str = ""
    # Tracing → Langfuse (observability._exporter). Ignored when
    # OTEL_EXPORTER_OTLP_ENDPOINT is set. trace_content=True also sends prompts and
    # completions to the tracing backend — off by default, they may contain user data.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    trace_content: bool = False

    # Input guardrail (guardrail.py): "block" rejects with 400, "log" only counts
    # (smartroute_guardrail_total) so you can measure false positives, "off" skips it.
    guardrail: str = "block"
    guardrail_max_chars: int = 100_000


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
if settings.togetherai_api_key:
    os.environ.setdefault("TOGETHERAI_API_KEY", settings.togetherai_api_key)
if settings.openrouter_api_key:
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
