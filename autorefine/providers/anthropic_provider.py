"""Anthropic Claude LLM provider.

Uses the official ``anthropic`` Python package.  The system prompt is
passed as a dedicated ``system`` parameter (not as a message), matching
Anthropic's API design.

Features:
- System prompt handled as a first-class API parameter
- Tool use support (pass ``tools=`` and ``tool_choice=``)
- Accurate token counts from the API response
- Per-model pricing via the ``PRICING`` dict
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from autorefine.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
)
from autorefine.models import Message
from autorefine.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger("autorefine.provider.anthropic")


def _classify_anthropic_error(exc: Exception, provider: str = "anthropic") -> ProviderError:
    msg = str(exc)
    status = getattr(exc, "status_code", None)
    if status in (401, 403) or "auth" in msg.lower() or "api key" in msg.lower():
        return ProviderAuthError(msg, provider=provider)
    if status == 429 or "rate" in msg.lower():
        return ProviderRateLimitError(msg, provider=provider)
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return ProviderNetworkError(msg, provider=provider)
    exc_type = type(exc).__name__
    if any(k in exc_type for k in ("Timeout", "Connect", "Network")):
        return ProviderNetworkError(msg, provider=provider)
    return ProviderError(msg, provider=provider, status_code=status)

# Per-1M-token costs: (input_usd, output_usd)
PRICING: dict[str, tuple[float, float]] = {
    # Claude 4.x family
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-6-20260415": (15.0, 75.0),
    # Claude 4.5
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    # Claude 3.5 family
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    # Claude 3 family
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
}

# Fallback — roughly Sonnet pricing
_DEFAULT_PRICING: tuple[float, float] = (3.0, 15.0)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude API.

    Args:
        api_key: Anthropic API key.
        model: Default model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        max_tokens: Default maximum output tokens per request.
        **client_kwargs: Forwarded to ``anthropic.Anthropic()``.

    Usage::

        provider = AnthropicProvider(api_key="sk-ant-...", model="claude-sonnet-4-20250514")

        # Simple one-shot
        resp = provider.complete("You are helpful.", "What is 2+2?")

        # Multi-turn
        resp = provider.chat("You are helpful.", messages)

        # Tool use
        resp = provider.chat(
            "You are helpful.",
            messages,
            tools=[{"name": "get_weather", "description": "...", "input_schema": {...}}],
        )
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        **client_kwargs: Any,
    ) -> None:
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Install with: pip install autorefine[anthropic]"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key, **client_kwargs)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key, **client_kwargs)
        self._model = model
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a chat completion to Claude.

        The ``system_prompt`` is passed via Anthropic's dedicated ``system``
        parameter rather than as a message, which gives Claude clearer
        context separation.

        Supports all Anthropic parameters via ``**kwargs``, including:

        - ``temperature``, ``top_p``, ``top_k``
        - ``max_tokens`` — override the default for this call
        - ``tools`` / ``tool_choice`` — for tool use
        - ``model`` — override the default model for this call
        """
        call_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "messages": self._to_dicts(messages),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt
        call_kwargs.update(kwargs)

        try:
            response = self._client.messages.create(**call_kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_anthropic_error(exc) from exc

        # Extract text from content blocks
        text_parts: list[str] = []
        tool_calls: list[Any] = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return ProviderResponse(
            text="".join(text_parts),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            finish_reason=response.stop_reason or "",
            tool_calls=tool_calls,
            raw=response,
        )

    def stream(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a chat completion from Claude, yielding text chunks."""
        call_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "messages": self._to_dicts(messages),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt
        call_kwargs.update(kwargs)

        try:
            with self._client.messages.stream(**call_kwargs) as stream:
                yield from stream.text_stream
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_anthropic_error(exc) from exc

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD using per-model pricing."""
        prices = self._resolve_pricing()
        return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_pricing(self) -> tuple[float, float]:
        """Look up pricing for the current model, with prefix fallback."""
        if self._model in PRICING:
            return PRICING[self._model]
        for key in sorted(PRICING, key=len, reverse=True):
            if self._model.startswith(key):
                return PRICING[key]
        return _DEFAULT_PRICING

    # ── Native async methods ─────────────────────────────────────────

    async def async_chat(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> ProviderResponse:
        call_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "messages": self._to_dicts(messages),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt
        call_kwargs.update(kwargs)

        try:
            response = await self._async_client.messages.create(**call_kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_anthropic_error(exc) from exc

        text_parts, tool_calls = [], []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        return ProviderResponse(
            text="".join(text_parts), input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens, model=response.model,
            finish_reason=response.stop_reason or "", tool_calls=tool_calls, raw=response,
        )

    async def async_stream(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        call_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "messages": self._to_dicts(messages),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt
        call_kwargs.update(kwargs)

        try:
            async with self._async_client.messages.stream(**call_kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_anthropic_error(exc) from exc
