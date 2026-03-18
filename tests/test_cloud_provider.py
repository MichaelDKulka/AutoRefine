"""Tests for the CloudProvider and key routing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from autorefine.exceptions import CloudAuthError, ProviderNetworkError, ProviderRateLimitError, SpendCapExceeded
from autorefine.models import Message, MessageRole
from autorefine.providers import get_provider
from autorefine.providers.cloud_provider import CloudProvider, _classify_error


# ── Key routing ──────────────────────────────────────────────────────

def test_ar_live_key_detected():
    provider = get_provider(api_key="ar_live_abc123def456", model="gpt-4o")
    assert isinstance(provider, CloudProvider)
    assert provider.name == "cloud"


def test_ar_test_key_detected():
    provider = get_provider(api_key="ar_test_abc123def456", model="gpt-4o")
    assert isinstance(provider, CloudProvider)


def test_regular_key_unchanged():
    """sk-... keys should NOT return CloudProvider."""
    provider = get_provider(api_key="sk-fake", model="gpt-4o")
    assert not isinstance(provider, CloudProvider)
    assert provider.name == "openai"


def test_anthropic_key_unchanged():
    """Anthropic keys should NOT return CloudProvider."""
    provider = get_provider(api_key="sk-ant-fake", model="claude-sonnet-4-20250514")
    assert not isinstance(provider, CloudProvider)
    assert provider.name == "anthropic"


# ── Error classification ─────────────────────────────────────────────

def test_auth_error_on_401():
    exc = _classify_error(401, "Invalid key")
    assert isinstance(exc, CloudAuthError)


def test_auth_error_on_403():
    exc = _classify_error(403, "Forbidden")
    assert isinstance(exc, CloudAuthError)


def test_spend_cap_on_402():
    exc = _classify_error(402, "Spend cap exceeded")
    assert isinstance(exc, SpendCapExceeded)


def test_rate_limit_on_429():
    exc = _classify_error(429, "Too many requests")
    assert isinstance(exc, ProviderRateLimitError)


def test_network_error_on_500():
    exc = _classify_error(500, "Internal server error")
    assert isinstance(exc, ProviderNetworkError)


def test_network_error_on_502():
    exc = _classify_error(502, "Bad gateway")
    assert isinstance(exc, ProviderNetworkError)


# ── Chat with mocked HTTP ────────────────────────────────────────────

def test_chat_routes_through_proxy():
    """Verify the request goes to the cloud URL."""
    provider = CloudProvider(api_key="ar_live_test123", model="gpt-4o", base_url="http://mock:9999")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Hello!",
        "input_tokens": 10,
        "output_tokens": 5,
        "model": "gpt-4o",
        "finish_reason": "stop",
    }

    with patch.object(provider._client, "post", return_value=mock_response) as mock_post:
        resp = provider.chat("Be helpful.", [Message(role=MessageRole.USER, content="Hi")])

        assert resp.text == "Hello!"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "/v1/chat"


def test_chat_auth_error():
    provider = CloudProvider(api_key="ar_live_bad", model="gpt-4o", base_url="http://mock:9999")
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Invalid API key"

    with patch.object(provider._client, "post", return_value=mock_response):
        with pytest.raises(CloudAuthError):
            provider.chat("Be helpful.", [Message(role=MessageRole.USER, content="Hi")])


def test_chat_spend_cap_error():
    provider = CloudProvider(api_key="ar_live_over", model="gpt-4o", base_url="http://mock:9999")
    mock_response = MagicMock()
    mock_response.status_code = 402
    mock_response.text = "Spend cap exceeded"

    with patch.object(provider._client, "post", return_value=mock_response):
        with pytest.raises(SpendCapExceeded):
            provider.chat("Be helpful.", [Message(role=MessageRole.USER, content="Hi")])


def test_chat_rate_limit_error():
    provider = CloudProvider(api_key="ar_live_fast", model="gpt-4o", base_url="http://mock:9999")
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limited"

    with patch.object(provider._client, "post", return_value=mock_response):
        with pytest.raises(ProviderRateLimitError):
            provider.chat("Be helpful.", [Message(role=MessageRole.USER, content="Hi")])


def test_network_error_on_timeout():
    import httpx
    provider = CloudProvider(api_key="ar_live_slow", model="gpt-4o", base_url="http://mock:9999")

    with patch.object(provider._client, "post", side_effect=httpx.TimeoutException("timed out")):
        with pytest.raises(ProviderNetworkError):
            provider.chat("Be helpful.", [Message(role=MessageRole.USER, content="Hi")])


# ── Estimate cost ────────────────────────────────────────────────────

def test_estimate_cost():
    provider = CloudProvider(api_key="ar_live_test", model="gpt-4o")
    cost = provider.estimate_cost(1000, 500)
    assert cost > 0


# ── Feedback submission ──────────────────────────────────────────────

def test_feedback_submission():
    provider = CloudProvider(api_key="ar_live_test", model="gpt-4o", base_url="http://mock:9999")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"feedback_id": "fb-123", "score": 1.0}

    with patch.object(provider._client, "post", return_value=mock_response):
        result = provider.submit_feedback("ix-123", "thumbs_up")
        assert result["feedback_id"] == "fb-123"
