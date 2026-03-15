"""Mistral AI LLM provider.

Uses the official ``mistralai`` Python package.

Features:
- Full Mistral API support via ``**kwargs``
- Per-model pricing via the ``PRICING`` dict
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from autorefine.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
)
from autorefine.models import Message
from autorefine.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger("autorefine.provider.mistral")


def _classify_mistral_error(exc: Exception, provider: str = "mistral") -> ProviderError:
    msg = str(exc)
    status = getattr(exc, "status_code", None)
    if status in (401, 403) or "auth" in msg.lower():
        return ProviderAuthError(msg, provider=provider)
    if status == 429 or "rate" in msg.lower():
        return ProviderRateLimitError(msg, provider=provider)
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return ProviderNetworkError(msg, provider=provider)
    return ProviderError(msg, provider=provider, status_code=status)

# Per-1M-token costs: (input_usd, output_usd)
PRICING: dict[str, tuple[float, float]] = {
    "mistral-tiny": (0.25, 0.25),
    "mistral-small-latest": (1.0, 3.0),
    "mistral-medium-latest": (2.7, 8.1),
    "mistral-large-latest": (4.0, 12.0),
    "codestral-latest": (1.0, 3.0),
}

_DEFAULT_PRICING: tuple[float, float] = (1.0, 3.0)


class MistralProvider(BaseProvider):
    """Provider for Mistral AI's API."""

    name = "mistral"

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        **client_kwargs: Any,
    ) -> None:
        try:
            from mistralai import Mistral  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "mistralai package is required. Install with: pip install autorefine[mistral]"
            ) from exc

        self._client = Mistral(api_key=api_key, **client_kwargs)
        self._model = model

    def chat(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> ProviderResponse:
        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._to_dicts(messages))

        try:
            response = self._client.chat.complete(
                model=kwargs.pop("model", self._model),
                messages=api_messages,
                **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_mistral_error(exc) from exc

        choice = response.choices[0]
        usage = response.usage
        return ProviderResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason or "",
            raw=response,
        )

    def stream(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._to_dicts(messages))

        try:
            stream_resp = self._client.chat.stream(
                model=kwargs.pop("model", self._model),
                messages=api_messages,
                **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify_mistral_error(exc) from exc

        for event in stream_resp:
            if event.data and event.data.choices:
                delta = event.data.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        prices = PRICING.get(self._model, _DEFAULT_PRICING)
        return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
