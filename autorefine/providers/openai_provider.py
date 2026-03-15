"""OpenAI (and OpenAI-compatible) LLM provider.

Supports any endpoint that implements the OpenAI chat completions API,
including Azure OpenAI, Together AI, Anyscale, vLLM, and LiteLLM proxies
via the ``base_url`` parameter.

Features:
- Function / tool calling (pass ``tools=`` and ``tool_choice=``)
- JSON mode (pass ``response_format={"type": "json_object"}``)
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
    ProviderResponseError,
)
from autorefine.models import Message
from autorefine.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger("autorefine.provider.openai")


def _classify_openai_error(exc: Exception, provider: str = "openai") -> ProviderError:
    """Map an OpenAI SDK exception to the correct AutoRefine exception."""
    msg = str(exc)
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 401 or status == 403 or "auth" in msg.lower() or "api key" in msg.lower():
        return ProviderAuthError(msg, provider=provider)
    if status == 429 or "rate" in msg.lower():
        return ProviderRateLimitError(msg, provider=provider)
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return ProviderNetworkError(msg, provider=provider)
    # Check for httpx/requests network errors
    exc_type = type(exc).__name__
    if any(k in exc_type for k in ("Timeout", "Connect", "Network", "DNS")):
        return ProviderNetworkError(msg, provider=provider)
    return ProviderError(msg, provider=provider, status_code=status)

# Per-1M-token costs: (input_usd, output_usd)
PRICING: dict[str, tuple[float, float]] = {
    # GPT-4.1 family
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    # GPT-4o family
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    # GPT-4 family
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    # GPT-3.5
    "gpt-3.5-turbo": (0.5, 1.5),
    # o-series reasoning
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o1-pro": (150.0, 600.0),
    "o3": (10.0, 40.0),
    "o3-mini": (1.1, 4.4),
    "o4-mini": (1.1, 4.4),
}

# Fallback for unknown models — roughly GPT-4o pricing
_DEFAULT_PRICING: tuple[float, float] = (2.5, 10.0)


class OpenAIProvider(BaseProvider):
    """Provider for the OpenAI chat completions API and any compatible endpoint.

    Args:
        api_key: OpenAI API key.
        model: Default model identifier (e.g. ``"gpt-4o"``).
        base_url: Override the API base URL for OpenAI-compatible services.
            Examples: ``"https://my-azure.openai.azure.com/v1"``,
            ``"http://localhost:8000/v1"`` (vLLM).
        **client_kwargs: Forwarded to ``openai.OpenAI()``.

    Usage::

        provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")

        # Simple one-shot
        resp = provider.complete("You are helpful.", "What is 2+2?")

        # Multi-turn
        resp = provider.chat("You are helpful.", messages)

        # JSON mode
        resp = provider.chat(
            "Return JSON.",
            messages,
            response_format={"type": "json_object"},
        )

        # Function calling
        resp = provider.chat(
            "You are helpful.",
            messages,
            tools=[{"type": "function", "function": {...}}],
            tool_choice="auto",
        )
    """

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: pip install autorefine[openai]"
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        kwargs.update(client_kwargs)
        self._client = openai.OpenAI(**kwargs)
        self._async_client = openai.AsyncOpenAI(**kwargs)
        self._model = model

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a chat completion request.

        Supports all OpenAI parameters via ``**kwargs``, including:

        - ``temperature``, ``top_p``, ``max_tokens``
        - ``response_format`` — e.g. ``{"type": "json_object"}`` for JSON mode
        - ``tools`` / ``tool_choice`` — for function calling
        - ``model`` — override the default model for this call
        """
        api_messages = self._build_messages(system_prompt, messages)
        model = kwargs.pop("model", self._model)

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=api_messages,
                **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_openai_error(exc) from exc

        try:
            choice = response.choices[0]
        except (IndexError, AttributeError) as exc:
            raise ProviderResponseError(
                f"Malformed response: no choices returned: {exc}",
                provider=self.name,
            ) from exc
        usage = response.usage

        # Extract tool calls if present
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return ProviderResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason or "",
            tool_calls=tool_calls,
            raw=response,
        )

    def stream(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a chat completion, yielding text chunks."""
        api_messages = self._build_messages(system_prompt, messages)
        model = kwargs.pop("model", self._model)

        try:
            stream_resp = self._client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True,
                **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_openai_error(exc) from exc

        for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD using per-model pricing."""
        prices = self._resolve_pricing()
        return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> list[dict[str, str]]:
        """Assemble the final message list with optional system prompt."""
        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._to_dicts(messages))
        return api_messages

    def _resolve_pricing(self) -> tuple[float, float]:
        """Look up pricing for the current model, falling back to defaults.

        Handles partial model name matches so that ``gpt-4o-2024-08-06``
        resolves to the ``gpt-4o`` entry.
        """
        if self._model in PRICING:
            return PRICING[self._model]
        # Try prefix matching for dated model snapshots
        for key in sorted(PRICING, key=len, reverse=True):
            if self._model.startswith(key):
                return PRICING[key]
        return _DEFAULT_PRICING

    # ── Native async methods ─────────────────────────────────────────

    async def async_chat(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> ProviderResponse:
        api_messages = self._build_messages(system_prompt, messages)
        model = kwargs.pop("model", self._model)
        try:
            response = await self._async_client.chat.completions.create(
                model=model, messages=api_messages, **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_openai_error(exc) from exc

        try:
            choice = response.choices[0]
        except (IndexError, AttributeError) as exc:
            raise ProviderResponseError(f"Malformed response: {exc}", provider=self.name) from exc

        usage = response.usage
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in choice.message.tool_calls]

        return ProviderResponse(
            text=choice.message.content or "", input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0, model=response.model,
            finish_reason=choice.finish_reason or "", tool_calls=tool_calls, raw=response,
        )

    async def async_stream(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        api_messages = self._build_messages(system_prompt, messages)
        model = kwargs.pop("model", self._model)
        try:
            stream_resp = await self._async_client.chat.completions.create(
                model=model, messages=api_messages, stream=True, **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_openai_error(exc) from exc

        async for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
