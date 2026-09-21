"""
SmartRoute Router: difficulty-aware LLM cascade.

Tiers:
  EASY  → cheap model directly (no verifier)
  MEDIUM → cheap → verifier ≥4 → return cheap; <4 → frontier
  HARD  → frontier directly

Which concrete models back "cheap" and "frontier" comes from the catalog
(models.yaml), not from this module. Swapping providers is a catalog edit.
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

from smartroute.cache import NoOpCache, PromptCache, get_cache
from typing import TYPE_CHECKING

from smartroute.catalog import Catalog, ModelSpec, get_catalog
from smartroute.classifier import DifficultyClassifier, DifficultyTier, get_classifier
from smartroute.config import settings
from smartroute.providers import ProviderError, call_model
from smartroute.verifier import CascadeVerifier

if TYPE_CHECKING:
    from smartroute.decision import DecisionLayer
    from smartroute.decomposer import Decomposer

logger = logging.getLogger(__name__)

_log_lock = asyncio.Lock()


@dataclass
class RoutingDecision:
    request_id: str
    prompt_hash: str                       # sha256 hex digest (no raw prompt stored)
    difficulty_tier: str                   # "EASY" | "MEDIUM" | "HARD"
    difficulty_score: float
    cascade_path: list[str] = field(default_factory=list)
    verifier_score: int | None = None
    verifier_correct: bool | None = None   # set by benchmark harness for MMLU
    cheap_response: str = ""               # raw cheap model output (populated when verifier runs)
    final_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    escalated: bool = False
    classifier_backend: str = "keyword"    # "keyword" | "deberta"
    extraction_failed: bool = False
    cache_hit: bool = False
    cache_similarity: float | None = None  # 1.0 for exact-match; future use for pgvector
    error: str | None = None

    # Two-layer decision fields (dispatch="decision")
    route_path: str = ""                   # "cheap" | "middle" | "frontier" | "decompose"
    gate_reason: str = ""                  # why the decision layer chose that path
    p_lookup: float | None = None
    p_decompose: float | None = None
    projected_cost_usd: float = 0.0
    subtask_count: int = 0


def _hash_prompt(messages: list[dict]) -> str:
    """SHA-256 of the full messages payload. No prompt text stored in logs."""
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_kwargs(backend: str) -> dict:
    base = {"ttl_seconds": settings.cache_ttl_seconds}
    if backend == "memory":
        base["max_size"] = settings.cache_max_size
    if backend == "pgvector":
        base["similarity_threshold"] = settings.cache_similarity_threshold
    return base


async def _append_decision(path: Path, decision: RoutingDecision) -> None:
    """Append a RoutingDecision as JSONL. Thread-safe via asyncio.Lock."""
    async with _log_lock:
        with path.open("a") as f:
            f.write(json.dumps(asdict(decision)) + "\n")


class Router:
    """
    Routes LLM requests based on classifier-determined difficulty.

    Args:
        classifier: DifficultyClassifier instance (default: keyword)
        verifier: CascadeVerifier instance (default: new instance)
        log_path: Path for RoutingDecision JSONL log (None = no logging)
    """

    def __init__(
        self,
        classifier: DifficultyClassifier | None = None,
        verifier: CascadeVerifier | None = None,
        cache: PromptCache | None = None,
        log_path: Path | None = None,
        verifier_enabled: bool = True,
        catalog: Catalog | None = None,
        dispatch: str = "decision",
        decision_layer: "DecisionLayer | None" = None,
        decomposer: "Decomposer | None" = None,
    ) -> None:
        if dispatch not in ("decision", "cascade"):
            raise ValueError(f"dispatch must be 'decision' or 'cascade', got {dispatch!r}")
        self._dispatch_mode = dispatch
        self._decision_layer = decision_layer
        self._decomposer = decomposer
        self._catalog = catalog or get_catalog()
        self._classifier = classifier or get_classifier(settings.classifier)
        self._verifier = verifier or CascadeVerifier()
        self._cache = cache if cache is not None else get_cache(
            settings.cache_backend,
            **_cache_kwargs(settings.cache_backend),
        )
        self._log_path = log_path
        self._verifier_enabled = verifier_enabled

        # Ordered candidate lists per tier. Today each tier holds one model, but the
        # sequence shape preserves the "try every candidate in a tier before
        # escalating" rule, and is what layer 2 will fan sub-tasks across.
        self._cheap_sequence = self._tier_sequence("cheap")
        self._frontier_sequence = self._tier_sequence("frontier")

    def _tier_sequence(self, role: str) -> list[ModelSpec]:
        return [m for m in self._catalog.all() if m.role == role]

    @property
    def decision_layer(self) -> "DecisionLayer":
        """Lazily built so cascade-mode routers never load the embedding model."""
        if self._decision_layer is None:
            from smartroute.decision import DecisionLayer

            self._decision_layer = DecisionLayer(catalog=self._catalog)
        return self._decision_layer

    @property
    def decomposer(self) -> "Decomposer":
        if self._decomposer is None:
            from smartroute.decomposer import Decomposer

            self._decomposer = Decomposer(catalog=self._catalog)
        return self._decomposer

    async def route_async(
        self,
        messages: list[dict],
        request_id: str | None = None,
    ) -> tuple[str, RoutingDecision]:
        """
        Route messages to the appropriate model.

        Returns:
            (response_content, RoutingDecision)

        Raises:
            ProviderError: if all frontier models fail
        """
        t0 = time.perf_counter()
        prompt_text = _extract_last_user_message(messages)
        rid = request_id or _hash_prompt(messages)[:16]

        # Cache lookup — must happen before classifier to maximize latency savings
        hit = await self._cache.get(prompt_text)
        if hit is not None:
            logger.debug("Cache hit (similarity=%.3f) for request %s", hit.similarity, rid)
            decision = RoutingDecision(
                request_id=rid,
                prompt_hash=_hash_prompt(messages),
                difficulty_tier="CACHED",
                difficulty_score=1.0,
                cascade_path=["cache"],
                final_model="cache",
                latency_ms=(time.perf_counter() - t0) * 1000,
                classifier_backend=self._classifier.backend(),
                cache_hit=True,
                cache_similarity=hit.similarity,
            )
            if self._log_path:
                await _append_decision(self._log_path, decision)
            return hit.response, decision

        if self._dispatch_mode == "decision":
            return await self._route_by_decision(messages, prompt_text, rid, t0)

        tier, score = self._classifier.classify(messages)

        decision = RoutingDecision(
            request_id=rid,
            prompt_hash=_hash_prompt(messages),
            difficulty_tier=tier.value,
            difficulty_score=score,
            classifier_backend=self._classifier.backend(),
        )

        try:
            content = await self._dispatch(messages, tier, decision)
        except Exception as e:
            decision.error = str(e)
            decision.latency_ms = (time.perf_counter() - t0) * 1000
            if self._log_path:
                await _append_decision(self._log_path, decision)
            raise

        decision.latency_ms = (time.perf_counter() - t0) * 1000

        # Write to cache on success (errors don't reach here — they're re-raised above)
        await self._cache.put(prompt_text, content)

        if self._log_path:
            await _append_decision(self._log_path, decision)

        return content, decision

    async def _route_by_decision(
        self,
        messages: list[dict],
        prompt_text: str,
        rid: str,
        t0: float,
    ) -> tuple[str, RoutingDecision]:
        """
        Two-layer dispatch: triage -> gate -> (cheap | middle | frontier | decompose).

        The frontier tier is never a default destination here; it is reached only when
        the gate's arithmetic prefers it to splitting, or via the decomposer's bail-out.
        """
        route = self.decision_layer.decide(prompt_text)

        decision = RoutingDecision(
            request_id=rid,
            prompt_hash=_hash_prompt(messages),
            # Kept populated so the Postgres schema, Prometheus labels and the
            # X-SmartRoute-Meta header contract keep working unchanged.
            difficulty_tier=route.path.upper(),
            difficulty_score=max(route.signals.p_lookup, route.signals.p_decompose),
            classifier_backend=route.signals.backend,
            route_path=route.path,
            gate_reason=route.reason,
            p_lookup=route.signals.p_lookup,
            p_decompose=route.signals.p_decompose,
            projected_cost_usd=route.projected_cost_usd,
        )

        try:
            if route.path == "decompose":
                result = await self.decomposer.run(prompt_text, messages)
                content = result.content
                decision.cascade_path = result.models_used
                decision.final_model = result.models_used[-1] if result.models_used else ""
                decision.input_tokens = result.input_tokens
                decision.output_tokens = result.output_tokens
                decision.estimated_cost_usd = result.cost_usd
                decision.subtask_count = len(result.subtasks)
                decision.gate_reason = f"{route.reason} | layer2: {result.reason}"
                decision.escalated = result.route == "bailout-frontier"
            else:
                content = await self._call_tier(route.path, messages, decision)
        except Exception as exc:
            decision.error = str(exc)
            decision.latency_ms = (time.perf_counter() - t0) * 1000
            if self._log_path:
                await _append_decision(self._log_path, decision)
            raise

        decision.latency_ms = (time.perf_counter() - t0) * 1000
        await self._cache.put(prompt_text, content)
        if self._log_path:
            await _append_decision(self._log_path, decision)
        return content, decision

    async def _call_tier(
        self, role: str, messages: list[dict], decision: RoutingDecision
    ) -> str:
        """Single call to a named catalog tier."""
        spec = self._catalog.by_role(role)
        decision.cascade_path.append(spec.id)
        resp = await call_model(
            spec.id, messages, timeout_s=settings.model_timeout_s, spec=spec
        )
        decision.final_model = spec.id
        decision.input_tokens += resp.input_tokens
        decision.output_tokens += resp.output_tokens
        decision.estimated_cost_usd += resp.estimated_cost_usd
        return resp.content

    async def _dispatch(
        self,
        messages: list[dict],
        tier: DifficultyTier,
        decision: RoutingDecision,
    ) -> str:
        if tier == DifficultyTier.HARD:
            return await self._call_frontier(messages, decision)

        if tier == DifficultyTier.EASY:
            return await self._call_cheap(messages, decision, use_verifier=False)

        # MEDIUM — skip verifier in routing-only mode
        return await self._call_cheap(messages, decision, use_verifier=self._verifier_enabled)

    async def _call_cheap(
        self,
        messages: list[dict],
        decision: RoutingDecision,
        use_verifier: bool,
    ) -> str:
        """Try cheap models in sequence. Falls through to frontier if all fail."""
        last_error: Exception | None = None

        for spec in self._cheap_sequence:
            model = spec.id
            decision.cascade_path.append(model)
            try:
                resp = await call_model(
                    model, messages, timeout_s=settings.model_timeout_s, spec=spec
                )
            except ProviderError as e:
                logger.warning("Cheap model %s failed: %s — trying next", model, e)
                last_error = e
                continue

            # Empty / refusal detection
            content = resp.content.strip()
            if not content:
                logger.warning("Cheap model %s returned empty response — escalating", model)
                decision.escalated = True
                decision.cascade_path.append("frontier:empty_response")
                return await self._call_frontier(messages, decision)

            decision.input_tokens += resp.input_tokens
            decision.output_tokens += resp.output_tokens
            decision.estimated_cost_usd += resp.estimated_cost_usd

            if not use_verifier:
                # EASY path — accept immediately
                decision.final_model = model
                return content

            # MEDIUM path — run verifier
            decision.cheap_response = content  # store before possible escalation
            try:
                verifier_score = await self._verifier.score_async(
                    prompt=_extract_last_user_message(messages),
                    response=content,
                    cheap_model=model,
                )
            except ProviderError as e:
                logger.warning("Verifier raised ProviderError: %s — confidence=2, escalating", e)
                verifier_score = 2
            decision.verifier_score = verifier_score

            if verifier_score >= settings.verifier_confidence_threshold:
                decision.final_model = model
                return content
            else:
                logger.info(
                    "Verifier score %d < threshold %d on %s — escalating to frontier",
                    verifier_score,
                    settings.verifier_confidence_threshold,
                    model,
                )
                decision.escalated = True
                return await self._call_frontier(messages, decision)

        # All cheap models failed
        logger.warning("All cheap models failed — escalating to frontier. Last error: %s", last_error)
        decision.escalated = True
        decision.cascade_path.append("frontier:all_cheap_failed")
        return await self._call_frontier(messages, decision)

    async def _call_frontier(
        self,
        messages: list[dict],
        decision: RoutingDecision,
    ) -> str:
        """Try frontier models in sequence. Raises ProviderError if all fail."""
        last_error: Exception | None = None

        for spec in self._frontier_sequence:
            model = spec.id
            if model not in decision.cascade_path:
                decision.cascade_path.append(model)
            try:
                resp = await call_model(
                    model, messages, timeout_s=settings.model_timeout_s, spec=spec
                )
            except ProviderError as e:
                logger.warning("Frontier model %s failed: %s — trying next", model, e)
                last_error = e
                continue

            decision.final_model = model
            decision.input_tokens += resp.input_tokens
            decision.output_tokens += resp.output_tokens
            decision.estimated_cost_usd += resp.estimated_cost_usd
            return resp.content

        raise ProviderError(
            f"All frontier models failed. Last error: {last_error}"
        )

    def route(
        self,
        messages: list[dict],
        request_id: str | None = None,
    ) -> tuple[str, RoutingDecision]:
        """
        Synchronous wrapper for route_async.
        Always uses ThreadPoolExecutor — safe to call from any context,
        including from within a running asyncio event loop.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                self.route_async(messages, request_id=request_id),
            )
            return future.result()


def _extract_last_user_message(messages: list[dict]) -> str:
    """Extract last user message for the verifier prompt."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""
