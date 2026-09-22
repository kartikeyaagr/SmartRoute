"""
Provider layer unit tests.
All LiteLLM calls are mocked — no real API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from smartroute.catalog import CatalogError, ModelSpec, Price
from smartroute.providers import ModelResponse, ProviderError, call_model

# Prices are the catalog's job; these tests are about transport, so they pass an
# explicit spec rather than depending on whatever models.yaml happens to contain.
SPEC = ModelSpec(
    alias="test", id="gpt-4o", role="middle",
    price=Price(input_per_token=1e-6, output_per_token=2e-6),
)


def _make_litellm_response(content="Paris", model="gpt-4o", input_tokens=10, output_tokens=5):
    """Build a minimal mock litellm response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    response.model = model
    return response


@pytest.mark.asyncio
async def test_rate_limit_retry_success():
    """RateLimitError on first attempt → wait 1s → second attempt succeeds."""
    mock_response = _make_litellm_response()

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise litellm.RateLimitError(
                message="rate limited", llm_provider="openai", model="gpt-4o"
            )
        return mock_response

    with patch("smartroute.providers.acompletion", side_effect=mock_acompletion):
        with patch("smartroute.providers.asyncio.sleep", new_callable=AsyncMock):
            with patch("smartroute.providers.litellm.completion_cost", return_value=0.001):
                result = await call_model("gpt-4o", [{"role": "user", "content": "hi"}], spec=SPEC)

    assert call_count == 2
    assert isinstance(result, ModelResponse)
    assert result.content == "Paris"


@pytest.mark.asyncio
async def test_rate_limit_retry_exhausted():
    """RateLimitError on both attempts → raises ProviderError."""

    async def mock_acompletion(**kwargs):
        raise litellm.RateLimitError(
            message="rate limited", llm_provider="openai", model="gpt-4o"
        )

    with patch("smartroute.providers.acompletion", side_effect=mock_acompletion):
        with patch("smartroute.providers.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ProviderError, match="RateLimitError"):
                await call_model("gpt-4o", [{"role": "user", "content": "hi"}], spec=SPEC)


@pytest.mark.asyncio
async def test_auth_failure_raises():
    """AuthenticationError → raises ProviderError immediately (no retry)."""
    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        raise litellm.AuthenticationError(
            message="invalid key", llm_provider="openai", model="gpt-4o"
        )

    with patch("smartroute.providers.acompletion", side_effect=mock_acompletion):
        with pytest.raises(ProviderError, match="AuthenticationError"):
            await call_model("gpt-4o", [{"role": "user", "content": "hi"}], spec=SPEC)

    assert call_count == 1  # no retry on auth failures


@pytest.mark.asyncio
async def test_unknown_model_raises_instead_of_costing_zero():
    """
    A model absent from the catalog must fail loudly.

    This replaces test_missing_pricing_defaults_zero, which asserted the opposite.
    Recording $0.00 for an unpriced model silently corrupts the cost figures that are
    the entire point of a cost-optimising router, so it is now an error.
    """
    async def mock_acompletion(**kwargs):
        return _make_litellm_response()

    with patch("smartroute.providers.acompletion", side_effect=mock_acompletion):
        with pytest.raises(CatalogError, match="unknown model"):
            await call_model("some-unknown-model", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_cost_comes_from_the_spec_not_the_registry():
    """Catalog price is authoritative even when litellm would price it differently."""
    async def mock_acompletion(**kwargs):
        return _make_litellm_response(input_tokens=1000, output_tokens=500)

    with patch("smartroute.providers.acompletion", side_effect=mock_acompletion):
        with patch("smartroute.providers.litellm.completion_cost", return_value=0.0):
            result = await call_model("gpt-4o", [{"role": "user", "content": "hi"}], spec=SPEC)

    # 1000 * 1e-6 + 500 * 2e-6
    assert result.estimated_cost_usd == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_extra_params_are_forwarded_to_the_provider():
    """temperature/max_tokens/logprobs used to be accepted and silently dropped."""
    seen = {}

    async def mock_acompletion(**kwargs):
        seen.update(kwargs)
        return _make_litellm_response()

    with patch("smartroute.providers.acompletion", side_effect=mock_acompletion):
        await call_model(
            "gpt-4o", [{"role": "user", "content": "hi"}],
            spec=SPEC, temperature=0, max_tokens=64, logprobs=True,
        )

    assert seen["temperature"] == 0
    assert seen["max_tokens"] == 64
    assert seen["logprobs"] is True
