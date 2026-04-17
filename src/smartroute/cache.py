"""
Pluggable prompt cache for SmartRoute.

Cache backends:
  NoOpCache        — always misses; default when CACHE_BACKEND=none
  InMemoryLRUCache — exact-match cache keyed on normalized prompt hash; CACHE_BACKEND=memory

Usage in Router:
  Before the classifier: check cache → on hit, return stored response immediately.
  After a successful LLM call: write (prompt, response, decision) to cache.

Semantic (pgvector) cache is a future backend (CACHE_BACKEND=pgvector).
cache_similarity is reserved in RoutingDecision for that use case.
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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

    def _key(self, prompt: str) -> str:
        normalized = prompt.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    async def get(self, prompt: str) -> CacheHit | None:
        import time
        key = self._key(prompt)
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
        import time
        key = self._key(prompt)

        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (response, time.monotonic())

        # Evict oldest entries beyond max_size
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)


def get_cache(backend: str, **kwargs) -> PromptCache:
    """Factory: returns the appropriate PromptCache implementation."""
    if backend == "memory":
        return InMemoryLRUCache(**kwargs)
    if backend == "none":
        return NoOpCache()
    raise ValueError(f"Unknown cache backend: {backend!r}. Valid options: 'none', 'memory', 'pgvector'")
