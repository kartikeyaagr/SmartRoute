"""
Provider layer unit tests.
All LiteLLM calls are mocked — no real API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from smartroute.providers import ModelResponse, ProviderError, call_model


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
                result = await call_model("gpt-4o", [{"role": "user", "content": "hi"}])

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
                await call_model("gpt-4o", [{"role": "user", "content": "hi"}])


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
            await call_model("gpt-4o", [{"role": "user", "content": "hi"}])

    assert call_count == 1  # no retry on auth failures


@pytest.mark.asyncio
async def test_missing_pricing_defaults_zero():
    """If LiteLLM has no pricing for model → cost=0.0, no exception raised."""
    mock_response = _make_litellm_response()

    async def mock_acompletion(**kwargs):
        return mock_response

    def mock_cost(**kwargs):
        raise Exception("no pricing data for this model")

    with patch("smartroute.providers.acompletion", side_effect=mock_acompletion):
        with patch("smartroute.providers.litellm.completion_cost", side_effect=mock_cost):
            result = await call_model("some-unknown-model", [{"role": "user", "content": "hi"}])

    assert result.estimated_cost_usd == 0.0
    assert result.content == "Paris"
