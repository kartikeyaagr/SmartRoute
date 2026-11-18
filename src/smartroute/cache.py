"""
Pluggable prompt cache for SmartRoute.

Cache backends:
  NoOpCache        — always misses; default when CACHE_BACKEND=none
  InMemoryLRUCache — exact-match LRU; CACHE_BACKEND=memory
  RedisCache       — exact-match, shared across replicas; CACHE_BACKEND=redis
  PgvectorCache    — semantic similarity via pgvector + sentence-transformers; CACHE_BACKEND=pgvector

Usage in Router:
  Before the classifier: check cache → on hit, return stored response immediately.
  After a successful LLM call: write (prompt, response) to cache.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    response: str
    similarity: float  # 1.0 for exact-match; 0.0–1.0 for semantic hits (pgvector)


@runtime_checkable
class PromptCache(Protocol):
    """Interface all cache backends must satisfy."""

    async def get(self, prompt: str) -> CacheHit | None:
        """Return a cached response, or None on miss."""
        ...

    async def put(self, prompt: str, response: str) -> None:
        """Store a prompt → response mapping."""
        ...


def exact_key(prompt: str) -> str:
    """SHA-256 of the normalized prompt — the key every exact-match backend shares."""
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()


class NoOpCache:
    """Always misses. Default when CACHE_BACKEND=none — zero behavior change."""

    async def get(self, prompt: str) -> CacheHit | None:
        return None

    async def put(self, prompt: str, response: str) -> None:
        pass


class InMemoryLRUCache:
    """
    Exact-match LRU cache. Keyed on SHA-256 of the normalized prompt.

    Not semantic — two prompts that mean the same thing but differ in whitespace
    or capitalization will miss each other. The pgvector backend handles semantic
    similarity. This backend validates the cache seam end-to-end with zero infra.

    Thread-safety: asyncio single-threaded. No lock needed for the dict itself.
    OrderedDict preserves insertion order for LRU eviction.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()  # key → (response, timestamp)

    async def get(self, prompt: str) -> CacheHit | None:
        key = exact_key(prompt)
        if key not in self._store:
            return None

        response, stored_at = self._store[key]
        if time.monotonic() - stored_at > self._ttl:
            del self._store[key]
            return None

        # Move to end (most recently used)
        self._store.move_to_end(key)
        return CacheHit(response=response, similarity=1.0)

    async def put(self, prompt: str, response: str) -> None:
        key = exact_key(prompt)

        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (response, time.monotonic())

        # Evict oldest entries beyond max_size
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)


class RedisCache:
    """
    Exact-match cache in Redis, shared by every replica.

    InMemoryLRUCache lives inside one process, so with N pods behind a load balancer
    each pod has its own cache and the hit rate drops ~N-fold. Redis is the one
    place they all read and write. Eviction is Redis's job (TTL via SET EX, plus
    `maxmemory-policy allkeys-lru` on the server), so there is no max_size here.

    Same failure contract as PgvectorCache: a Redis outage degrades to a miss,
    never to a failed request.
    """

    def __init__(self, url: str, ttl_seconds: float = 3600.0, client=None) -> None:
        if client is None:
            import redis.asyncio as redis  # [server] extra

            client = redis.from_url(url, decode_responses=True, socket_timeout=0.5)
        self._redis = client
        self._ttl = max(1, int(ttl_seconds))

    async def get(self, prompt: str) -> CacheHit | None:
        try:
            response = await self._redis.get("sr:" + exact_key(prompt))
        except Exception as exc:
            logger.warning("RedisCache.get failed: %s", exc)
            return None
        return CacheHit(response=response, similarity=1.0) if response is not None else None

    async def put(self, prompt: str, response: str) -> None:
        try:
            await self._redis.set("sr:" + exact_key(prompt), response, ex=self._ttl)
        except Exception as exc:
            logger.warning("RedisCache.put failed: %s", exc)


class PgvectorCache:
    """
    Semantic cache backed by PostgreSQL + pgvector.

    Embeds prompts with sentence-transformers (all-MiniLM-L6-v2, 90MB, local — no data
    sent to external APIs). Cosine similarity lookup: if the closest stored prompt scores
    above `similarity_threshold`, return its cached response.

    Per-tenant isolation: set `tenant_id` so cache entries never cross tenant boundaries.
    CACHE_SIMILARITY_THRESHOLD controls the hit/miss boundary (default 0.95).

    Requires: asyncpg pool initialised via db.init_pool(), pgvector extension installed.
    Falls back to miss on any DB error — never blocks the request path.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS prompt_cache (
        id          BIGSERIAL PRIMARY KEY,
        tenant_id   TEXT            NOT NULL DEFAULT 'default',
        prompt_hash TEXT            NOT NULL,
        prompt      TEXT            NOT NULL,
        response    TEXT            NOT NULL,
        embedding   vector(384),
        created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        expires_at  TIMESTAMPTZ     NOT NULL
    )
    """

    _CREATE_INDEX = """
    CREATE INDEX IF NOT EXISTS prompt_cache_embedding_idx
    ON prompt_cache USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
    """

    _UPSERT = """
    INSERT INTO prompt_cache (tenant_id, prompt_hash, prompt, response, embedding, expires_at)
    VALUES ($1, $2, $3, $4, $5::vector, NOW() + $6 * INTERVAL '1 second')
    ON CONFLICT DO NOTHING
    """

    _LOOKUP = """
    SELECT response, 1 - (embedding <=> $1::vector) AS similarity
    FROM prompt_cache
    WHERE tenant_id = $2 AND expires_at > NOW()
    ORDER BY embedding <=> $1::vector
    LIMIT 1
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: float = 3600.0,
        tenant_id: str = "default",
    ) -> None:
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._tenant_id = tenant_id
        self._encoder = None  # lazy-loaded on first use

    def _encode(self, text: str) -> list[float]:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._MODEL_NAME)
        return self._encoder.encode(text, normalize_embeddings=True).tolist()

    async def _ensure_schema(self, conn) -> None:
        await conn.execute(self._CREATE_TABLE)
        try:
            await conn.execute(self._CREATE_INDEX)
        except Exception:
            pass  # index creation fails gracefully if vectors not yet populated

    async def get(self, prompt: str) -> "CacheHit | None":
        from smartroute import db as _db
        if _db._pool is None:
            return None
        try:
            embedding = self._encode(prompt)
            async with _db._pool.acquire() as conn:
                await self._ensure_schema(conn)
                row = await conn.fetchrow(self._LOOKUP, embedding, self._tenant_id)
            if row and row["similarity"] >= self._threshold:
                return CacheHit(response=row["response"], similarity=float(row["similarity"]))
        except Exception as exc:
            logger.warning("PgvectorCache.get failed: %s", exc)
        return None

    async def put(self, prompt: str, response: str) -> None:
        from smartroute import db as _db
        if _db._pool is None:
            return
        try:
            embedding = self._encode(prompt)
            prompt_hash = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
            async with _db._pool.acquire() as conn:
                await self._ensure_schema(conn)
                await conn.execute(
                    self._UPSERT,
                    self._tenant_id,
                    prompt_hash,
                    prompt,
                    response,
                    embedding,
                    self._ttl,
                )
        except Exception as exc:
            logger.warning("PgvectorCache.put failed: %s", exc)


def get_cache(backend: str, **kwargs) -> PromptCache:
    """Factory: returns the appropriate PromptCache implementation."""
    if backend == "memory":
        return InMemoryLRUCache(**kwargs)
    if backend == "none":
        return NoOpCache()
    if backend == "redis":
        return RedisCache(**kwargs)
    if backend == "pgvector":
        return PgvectorCache(**kwargs)
    raise ValueError(f"Unknown cache backend: {backend!r}. Valid options: 'none', 'memory', 'redis', 'pgvector'")
