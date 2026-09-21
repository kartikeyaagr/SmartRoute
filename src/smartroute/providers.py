"""
Thin async wrapper around litellm.acompletion.
Tracks cost, latency, tokens. Handles retries and provider errors.

Pricing authority
-----------------
Cost is computed from the catalog (`ModelSpec.price`), not from
`litellm.completion_cost`. That function prices from the *response object's* model
string, which for `together_ai/...` frequently arrives without its provider prefix,
resolves to nothing, and returns 0.0 — which the old code then logged at DEBUG level
and recorded as $0.00. Silently reporting zero cost corrupts the only number this
project exists to produce.

`completion_cost` is still consulted, but demoted to a cross-check: a >5% divergence
from the catalog logs a WARNING so a stale catalog price surfaces instead of rotting.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

from typing import TYPE_CHECKING

from smartroute.catalog import CatalogError
from smartroute.config import settings

if TYPE_CHECKING:
    from smartroute.catalog import ModelSpec

logger = logging.getLogger(__name__)

_PRICE_DIVERGENCE_TOLERANCE = 0.05  # warn when catalog and registry disagree by >5%


# ---------------------------------------------------------------------------
# Lazy litellm
# ---------------------------------------------------------------------------
# Importing litellm costs ~1.6s, and `import smartroute` pulls this module in, so the
# import is deferred until the first real call. PEP 562 module __getattr__ keeps
# `smartroute.providers.acompletion` resolvable from outside — which is what
# mock.patch() needs to find before it can replace it. Internal callers go through
# _acompletion(), which reads the same module global, so a patched mock is honoured.

_LAZY_LITELLM_ATTRS = {"acompletion", "litellm"}


def __getattr__(name: str):
    if name in _LAZY_LITELLM_ATTRS:
        import litellm as _ll

        value = _ll.acompletion if name == "acompletion" else _ll
        globals()[name] = value  # cache: later lookups skip __getattr__ entirely
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _acompletion():
    """Module-global handle to litellm.acompletion, resolved once and patchable."""
    fn = globals().get("acompletion")
    if fn is None:
        from litellm import acompletion as fn  # noqa: PLC0415

        globals()["acompletion"] = fn
    return fn


def _litellm():
    """Lazy litellm handle — importing it eagerly costs ~1.6s of startup."""
    import litellm

    return litellm


def _cost_for(model, spec, input_tokens, output_tokens, response) -> float:
    """
    Cost from the catalog, cross-checked against litellm's own estimate.

    Raises CatalogError if `model` is unknown, rather than recording $0.00.
    """
    if spec is None:
        from smartroute.catalog import get_catalog

        spec = get_catalog().resolve(model)

    cost = spec.cost(input_tokens, output_tokens)

    try:
        registry_cost = _litellm().completion_cost(completion_response=response)
    except Exception:
        return cost
    if registry_cost and cost and abs(registry_cost - cost) / cost > _PRICE_DIVERGENCE_TOLERANCE:
        logger.warning(
            "price divergence for %s: catalog $%.8f vs litellm $%.8f (>%.0f%%) — "
            "the catalog price may be stale",
            model, cost, registry_cost, _PRICE_DIVERGENCE_TOLERANCE * 100,
        )
    return cost


class ProviderError(Exception):
    """Raised when all retry attempts for a model have failed."""


@dataclass
class ModelResponse:
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float


async def call_model(
    model: str,
    messages: list[dict],
    timeout_s: float | None = None,
    spec: "ModelSpec | None" = None,
    **params,
) -> ModelResponse:
    """
    Call a model via LiteLLM with retry on RateLimitError.

    Failure handling:
    - RateLimitError: retry once after 1s
    - AuthenticationError, ServiceUnavailableError: raise ProviderError immediately
    - Timeout: raise ProviderError

    Args:
        spec: catalog entry for `model`. When given it is the pricing authority.
            When omitted the catalog is consulted by model id; a model absent from
            the catalog raises rather than silently costing $0.00.
        **params: forwarded verbatim to the provider (temperature, max_tokens,
            logprobs, ...). Previously dropped on the floor, which made reproducible
            evaluation impossible — you cannot pin temperature=0 through a wrapper
            that discards it.

    Raises ProviderError on unrecoverable failures.
    """
    timeout_s = timeout_s or settings.model_timeout_s

    async def _attempt() -> ModelResponse:
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                _acompletion()(model=model, messages=messages, **params),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise ProviderError(f"Model {model} timed out after {timeout_s}s")

        latency_ms = (time.perf_counter() - t0) * 1000

        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        cost = _cost_for(model, spec, input_tokens, output_tokens, response)

        content = response.choices[0].message.content or ""

        return ModelResponse(
            model=model,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            latency_ms=latency_ms,
        )

    # First attempt, then up to 3 retries with exponential backoff on rate limits
    try:
        return await _attempt()
    except (ProviderError, CatalogError):
        raise
    except _litellm().RateLimitError:
        for attempt in range(1, 4):
            wait = 5.0 * attempt * attempt  # 5s, 20s, 45s
            logger.warning("RateLimitError on %s — retrying after %.0fs (attempt %d/3)", model, wait, attempt)
            await asyncio.sleep(wait)
            try:
                return await _attempt()
            except _litellm().RateLimitError:
                if attempt == 3:
                    raise ProviderError(f"RateLimitError on {model} after 3 retries")
            except (ProviderError, CatalogError):
                raise
            except Exception as e:
                raise ProviderError(f"Unexpected error on {model}: {e}") from e
        raise ProviderError(f"RateLimitError on {model} after 3 retries")  # unreachable but satisfies type checker
    except _litellm().AuthenticationError as e:
        raise ProviderError(f"AuthenticationError for {model}: {e}") from e
    except _litellm().ServiceUnavailableError as e:
        raise ProviderError(f"ServiceUnavailableError for {model}: {e}") from e
    except Exception as e:
        raise ProviderError(f"Unexpected error on {model}: {e}") from e
