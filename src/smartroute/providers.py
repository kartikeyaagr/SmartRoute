"""
Thin async wrapper around litellm.acompletion.
Tracks cost, latency, tokens. Handles retries and provider errors.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import litellm
from litellm import acompletion

from smartroute.config import settings

logger = logging.getLogger(__name__)

# Manual per-token pricing fallback for models LiteLLM doesn't price.
# Rates are (input_per_token_usd, output_per_token_usd).
_MANUAL_PRICING: dict[str, tuple[float, float]] = {
    "groq/llama-3.1-8b-instant":    (0.05 / 1_000_000, 0.08 / 1_000_000),
    "groq/llama-3.3-70b-versatile": (0.59 / 1_000_000, 0.79 / 1_000_000),
    "groq/qwen/qwen3-32b":           (0.29 / 1_000_000, 0.59 / 1_000_000),
}


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
) -> ModelResponse:
    """
    Call a model via LiteLLM with retry on RateLimitError.

    Failure handling:
    - RateLimitError: retry once after 1s
    - AuthenticationError, ServiceUnavailableError: raise ProviderError immediately
    - Timeout: raise ProviderError
    - Missing pricing: cost=0.0, log WARNING

    Raises ProviderError on unrecoverable failures.
    """
    timeout_s = timeout_s or settings.model_timeout_s

    async def _attempt() -> ModelResponse:
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                acompletion(model=model, messages=messages),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise ProviderError(f"Model {model} timed out after {timeout_s}s")

        latency_ms = (time.perf_counter() - t0) * 1000

        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        if cost == 0.0 and model in _MANUAL_PRICING:
            in_rate, out_rate = _MANUAL_PRICING[model]
            cost = input_tokens * in_rate + output_tokens * out_rate
            logger.debug("Manual pricing applied for %s: $%.6f", model, cost)
        elif cost == 0.0:
            logger.debug("No pricing data for model %s — cost recorded as $0.00", model)

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
    except ProviderError:
        raise
    except litellm.RateLimitError:
        for attempt in range(1, 4):
            wait = 5.0 * attempt * attempt  # 5s, 20s, 45s
            logger.warning("RateLimitError on %s — retrying after %.0fs (attempt %d/3)", model, wait, attempt)
            await asyncio.sleep(wait)
            try:
                return await _attempt()
            except litellm.RateLimitError:
                if attempt == 3:
                    raise ProviderError(f"RateLimitError on {model} after 3 retries")
            except ProviderError:
                raise
            except Exception as e:
                raise ProviderError(f"Unexpected error on {model}: {e}") from e
        raise ProviderError(f"RateLimitError on {model} after 3 retries")  # unreachable but satisfies type checker
    except litellm.AuthenticationError as e:
        raise ProviderError(f"AuthenticationError for {model}: {e}") from e
    except litellm.ServiceUnavailableError as e:
        raise ProviderError(f"ServiceUnavailableError for {model}: {e}") from e
    except Exception as e:
        raise ProviderError(f"Unexpected error on {model}: {e}") from e
