"""
Postgres persistence layer for SmartRoute.

Responsibilities:
  - Connection pool lifecycle (init / close via FastAPI lifespan)
  - Schema creation (routing_decisions table + pgvector extension)
  - Async insert of RoutingDecision records

When DATABASE_URL is empty the module is a no-op — the server runs without Postgres.
"""

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

_pool: "asyncpg.Pool | None" = None

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS routing_decisions (
    id              BIGSERIAL PRIMARY KEY,
    request_id      TEXT        NOT NULL,
    prompt_hash     TEXT        NOT NULL,
    difficulty_tier TEXT        NOT NULL,
    difficulty_score DOUBLE PRECISION NOT NULL,
    cascade_path    JSONB       NOT NULL DEFAULT '[]',
    verifier_score  SMALLINT,
    escalated       BOOLEAN     NOT NULL DEFAULT FALSE,
    final_model     TEXT        NOT NULL DEFAULT '',
    input_tokens    INTEGER     NOT NULL DEFAULT 0,
    output_tokens   INTEGER     NOT NULL DEFAULT 0,
    estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_ms      DOUBLE PRECISION NOT NULL DEFAULT 0,
    classifier_backend TEXT     NOT NULL DEFAULT 'keyword',
    cache_hit       BOOLEAN     NOT NULL DEFAULT FALSE,
    cache_similarity DOUBLE PRECISION,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_INSERT = """
INSERT INTO routing_decisions (
    request_id, prompt_hash, difficulty_tier, difficulty_score,
    cascade_path, verifier_score, escalated, final_model,
    input_tokens, output_tokens, estimated_cost_usd, latency_ms,
    classifier_backend, cache_hit, cache_similarity, error
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
"""


async def init_pool(database_url: str) -> None:
    """Create the connection pool and ensure schema exists."""
    global _pool
    import asyncpg  # deferred — not installed in library-only mode

    _pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
    async with _pool.acquire() as conn:
        # pgvector extension — best-effort (fails silently if not installed)
        try:
            await conn.execute(_CREATE_EXTENSION)
        except Exception as exc:
            logger.warning("pgvector extension not available: %s", exc)
        await conn.execute(_CREATE_TABLE)
    logger.info("Postgres pool initialised")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def insert_decision(decision) -> None:
    """Insert a RoutingDecision. Silent no-op when pool is not initialised."""
    if _pool is None:
        return
    d = asdict(decision)
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                _INSERT,
                d["request_id"],
                d["prompt_hash"],
                d["difficulty_tier"],
                d["difficulty_score"],
                json.dumps(d["cascade_path"]),
                d["verifier_score"],
                d["escalated"],
                d["final_model"],
                d["input_tokens"],
                d["output_tokens"],
                d["estimated_cost_usd"],
                d["latency_ms"],
                d["classifier_backend"],
                d["cache_hit"],
                d["cache_similarity"],
                d["error"],
            )
    except Exception as exc:
        # Never let a DB write failure kill the API response
        logger.error("Failed to insert routing_decision %s: %s", d["request_id"], exc)
