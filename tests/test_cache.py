"""
Tests for cache.py and Router cache integration.

All tests use mocked providers — no real API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from smartroute.cache import CacheHit, InMemoryLRUCache, NoOpCache, PgvectorCache, PromptCache, get_cache
from smartroute.classifier import DifficultyTier
from smartroute.providers import ModelResponse, ProviderError
from smartroute.router import Router


# ---------------------------------------------------------------------------
# NoOpCache
# ---------------------------------------------------------------------------

async def test_noop_cache_always_misses():
    cache = NoOpCache()
    assert await cache.get("any prompt") is None


async def test_noop_cache_put_is_silent():
    cache = NoOpCache()
    await cache.put("any prompt", "any response")  # should not raise
    assert await cache.get("any prompt") is None  # still misses


# ---------------------------------------------------------------------------
# InMemoryLRUCache — basic put/get
# ---------------------------------------------------------------------------

async def test_memory_cache_miss_on_empty():
    cache = InMemoryLRUCache()
    assert await cache.get("hello") is None


async def test_memory_cache_put_then_get():
    cache = InMemoryLRUCache()
    await cache.put("what is the capital of France?", "Paris")
    hit = await cache.get("what is the capital of France?")
    assert hit is not None
    assert hit.response == "Paris"
    assert hit.similarity == 1.0


async def test_memory_cache_case_and_whitespace_normalization():
    cache = InMemoryLRUCache()
    await cache.put("  What is 2+2?  ", "4")
    # Leading/trailing whitespace normalized; case normalized
    hit = await cache.get("what is 2+2?")
    assert hit is not None
    assert hit.response == "4"


async def test_memory_cache_different_prompts_dont_collide():
    cache = InMemoryLRUCache()
    await cache.put("prompt A", "response A")
    await cache.put("prompt B", "response B")
    assert (await cache.get("prompt A")).response == "response A"
    assert (await cache.get("prompt B")).response == "response B"


async def test_memory_cache_lru_eviction():
    cache = InMemoryLRUCache(max_size=2)
    await cache.put("prompt 1", "resp 1")
    await cache.put("prompt 2", "resp 2")
    await cache.put("prompt 3", "resp 3")  # evicts prompt 1 (oldest)
    assert await cache.get("prompt 1") is None
    assert (await cache.get("prompt 2")).response == "resp 2"
    assert (await cache.get("prompt 3")).response == "resp 3"


async def test_memory_cache_ttl_expiry(monkeypatch):
    import time as _time
    cache = InMemoryLRUCache(ttl_seconds=1.0)
    await cache.put("expires soon", "value")

    # Simulate time passing beyond TTL
    original_monotonic = _time.monotonic
    monkeypatch.setattr(_time, "monotonic", lambda: original_monotonic() + 2.0)

    assert await cache.get("expires soon") is None


# ---------------------------------------------------------------------------
# get_cache factory
# ---------------------------------------------------------------------------

def test_get_cache_none_returns_noop():
    cache = get_cache("none")
    assert isinstance(cache, NoOpCache)


def test_get_cache_memory_returns_lru():
    cache = get_cache("memory")
    assert isinstance(cache, InMemoryLRUCache)


def test_get_cache_pgvector_returns_pgvector():
    cache = get_cache("pgvector")
    assert isinstance(cache, PgvectorCache)


def test_get_cache_unknown_raises():
    with pytest.raises(ValueError, match="Unknown cache backend"):
        get_cache("redis")


def test_prompt_cache_protocol():
    assert isinstance(NoOpCache(), PromptCache)
    assert isinstance(InMemoryLRUCache(), PromptCache)
    assert isinstance(PgvectorCache(), PromptCache)


# ---------------------------------------------------------------------------
# PgvectorCache — no-pool behaviour (no real Postgres required)
# ---------------------------------------------------------------------------

async def test_pgvector_get_returns_none_when_no_pool():
    """PgvectorCache.get is a miss when DB pool is uninitialised."""
    import smartroute.db as db_module
    original = db_module._pool
    db_module._pool = None
    try:
        cache = PgvectorCache()
        result = await cache.get("any prompt")
        assert result is None
    finally:
        db_module._pool = original


async def test_pgvector_put_is_noop_when_no_pool():
    """PgvectorCache.put does nothing when DB pool is uninitialised."""
    import smartroute.db as db_module
    original = db_module._pool
    db_module._pool = None
    try:
        cache = PgvectorCache()
        await cache.put("any prompt", "any response")  # must not raise
    finally:
        db_module._pool = original


async def test_pgvector_get_returns_none_on_db_error(monkeypatch):
    """PgvectorCache.get returns None (miss) on any DB exception — never raises."""
    import smartroute.db as db_module

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetchrow = AsyncMock(side_effect=Exception("connection lost"))
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    original = db_module._pool
    db_module._pool = mock_pool

    # Mock out the encoder so we don't download a model
    cache = PgvectorCache()
    cache._encoder = MagicMock()
    cache._encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

    try:
        result = await cache.get("test prompt")
        assert result is None
    finally:
        db_module._pool = original


async def test_pgvector_below_threshold_is_miss(monkeypatch):
    """A similarity score below threshold must not be returned as a hit."""
    import smartroute.db as db_module

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"response": "cached", "similarity": 0.80})
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    original = db_module._pool
    db_module._pool = mock_pool

    cache = PgvectorCache(similarity_threshold=0.95)
    cache._encoder = MagicMock()
    cache._encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

    try:
        result = await cache.get("test prompt")
        assert result is None  # 0.80 < 0.95 threshold → miss
    finally:
        db_module._pool = original


async def test_pgvector_above_threshold_is_hit(monkeypatch):
    """A similarity score at/above threshold returns a CacheHit."""
    import smartroute.db as db_module

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"response": "cached answer", "similarity": 0.97})
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    original = db_module._pool
    db_module._pool = mock_pool

    cache = PgvectorCache(similarity_threshold=0.95)
    cache._encoder = MagicMock()
    cache._encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

    try:
        hit = await cache.get("test prompt")
        assert hit is not None
        assert hit.response == "cached answer"
        assert hit.similarity == pytest.approx(0.97)
    finally:
        db_module._pool = original


# ---------------------------------------------------------------------------
# Router cache integration
# ---------------------------------------------------------------------------

def _messages(text: str = "What is 2+2?") -> list[dict]:
    return [{"role": "user", "content": text}]


def _mock_response(content: str = "4", model: str = "groq/llama-3.1-8b-instant") -> ModelResponse:
    return ModelResponse(
        model=model,
        content=content,
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=0.0001,
        latency_ms=120.0,
    )


def test_router_default_uses_noop_cache():
    """Router() with default settings → NoOpCache (CACHE_BACKEND=none is the default)."""
    clf = MagicMock()
    clf.classify.return_value = (DifficultyTier.EASY, 0.8)
    clf.backend.return_value = "keyword"

    router = Router(classifier=clf)
    assert isinstance(router._cache, NoOpCache)


async def test_router_cache_miss_calls_providers():
    """Cache miss → classifier + providers run as normal."""
    clf = MagicMock()
    clf.classify.return_value = (DifficultyTier.EASY, 0.8)
    clf.backend.return_value = "keyword"

    cache = NoOpCache()  # always misses

    with MagicMock() as mock_call:
        from unittest.mock import patch
        with patch("smartroute.router.call_model", new=AsyncMock(return_value=_mock_response("42"))):
            router = Router(classifier=clf, cache=cache)
            content, decision = await router.route_async(_messages())

    assert content == "42"
    assert decision.cache_hit is False
    assert decision.cache_similarity is None


async def test_router_cache_hit_skips_providers():
    """Warm cache → return stored response immediately, no provider call."""
    clf = MagicMock()
    clf.classify.return_value = (DifficultyTier.EASY, 0.8)
    clf.backend.return_value = "keyword"

    cache = InMemoryLRUCache()
    await cache.put("What is 2+2?", "4")  # pre-warm

    with MagicMock():
        from unittest.mock import patch
        with patch("smartroute.router.call_model", new=AsyncMock(side_effect=AssertionError("providers should not be called on cache hit"))):
            router = Router(classifier=clf, cache=cache)
            content, decision = await router.route_async(_messages("What is 2+2?"))

    assert content == "4"
    assert decision.cache_hit is True
    assert decision.cache_similarity == 1.0
    assert decision.cascade_path == ["cache"]
    assert decision.final_model == "cache"
    assert decision.difficulty_tier == "CACHED"


async def test_router_writes_to_cache_after_llm_call():
    """Successful LLM call → response written to cache → second call is a hit."""
    clf = MagicMock()
    clf.classify.return_value = (DifficultyTier.EASY, 0.8)
    clf.backend.return_value = "keyword"

    cache = InMemoryLRUCache()

    from unittest.mock import patch
    with patch("smartroute.router.call_model", new=AsyncMock(return_value=_mock_response("Paris"))):
        router = Router(classifier=clf, cache=cache)
        content1, decision1 = await router.route_async(_messages("What is the capital of France?"))

    assert content1 == "Paris"
    assert decision1.cache_hit is False

    # Second call — same router, same cache — should hit
    with patch("smartroute.router.call_model", new=AsyncMock(side_effect=AssertionError("should be cache hit"))):
        content2, decision2 = await router.route_async(_messages("What is the capital of France?"))

    assert content2 == "Paris"
    assert decision2.cache_hit is True
