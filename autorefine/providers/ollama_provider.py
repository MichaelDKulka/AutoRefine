"""Ollama provider for locally-hosted models.

Communicates directly with Ollama's REST API via ``httpx`` — no SDK
dependency required.  All cost tracking returns ``0.0`` since local
models are free.

Features:
- Zero external dependencies beyond ``httpx`` (already a core dep)
- Configurable base URL (default ``http://localhost:11434``)
- Accurate token counts from Ollama's ``prompt_eval_count`` / ``eval_count``
- Streaming via Ollama's NDJSON stream endpoint
"""

from __future__ import annotations

import json as _json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from autorefine.exceptions import ProviderError, ProviderNetworkError
from autorefine.models import Message
from autorefine.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger("autorefine.provider.ollama")

# Ollama models run locally — cost is always zero.
PRICING: dict[str, tuple[float, float]] = {
    "llama3": (0.0, 0.0),
    "llama3.1": (0.0, 0.0),
    "llama3.2": (0.0, 0.0),
    "llama3.3": (0.0, 0.0),
    "gemma": (0.0, 0.0),
    "gemma2": (0.0, 0.0),
    "phi3": (0.0, 0.0),
    "phi4": (0.0, 0.0),
    "qwen2.5": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "mixtral": (0.0, 0.0),
    "codellama": (0.0, 0.0),
    "deepseek-coder": (0.0, 0.0),
    "deepseek-r1": (0.0, 0.0),
    "command-r": (0.0, 0.0),
}


class OllamaProvider(BaseProvider):
    """Provider for Ollama's REST API (locally-hosted models).

    Args:
        model: Default model name (e.g. ``"llama3.1"``).  Must be
            already pulled in Ollama (``ollama pull llama3.1``).
        base_url: Ollama server URL.  Defaults to ``http://localhost:11434``.
        timeout: Request timeout in seconds.  Defaults to 120 for large
            models that take a while to generate.
        **kwargs: Forwarded to ``httpx.Client()``.

    Usage::

        provider = OllamaProvider(model="llama3.1")

        # Simple one-shot
        resp = provider.complete("You are helpful.", "What is 2+2?")

        # Multi-turn
        resp = provider.chat("You are helpful.", messages)

        # Streaming
        for chunk in provider.stream("You are helpful.", messages):
            print(chunk, end="")
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout, **kwargs)
        self._async_client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout, **kwargs)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a chat request to Ollama's ``/api/chat`` endpoint."""
        api_messages = self._build_messages(system_prompt, messages)
        model = kwargs.pop("model", self._model)

        try:
            resp = self._client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": api_messages,
                    "stream": False,
                    **kwargs,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama returned {exc.response.status_code}: {exc.response.text[:200]}",
                provider=self.name,
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(str(exc), provider=self.name) from exc

        data = resp.json()
        message = data.get("message", {})

        return ProviderResponse(
            text=message.get("content", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=data.get("model", model),
            finish_reason=data.get("done_reason", "stop") if data.get("done") else "",
            raw=data,
        )

    def stream(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a chat completion from Ollama via NDJSON.

        Ollama's streaming endpoint returns one JSON object per line,
        each with ``message.content`` containing the next token(s).
        """
        api_messages = self._build_messages(system_prompt, messages)
        model = kwargs.pop("model", self._model)

        try:
            with self._client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": api_messages,
                    "stream": True,
                    **kwargs,
                },
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = _json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
        except ProviderError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama returned {exc.response.status_code}: {exc.response.text[:200]}",
                provider=self.name,
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(str(exc), provider=self.name) from exc

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local models are free — always returns ``0.0``."""
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> list[dict[str, str]]:
        """Assemble the message list with optional system prompt."""
        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._to_dicts(messages))
        return api_messages

    # ── Native async methods ─────────────────────────────────────────

    async def async_chat(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> ProviderResponse:
        api_messages = self._build_messages(system_prompt, messages)
        model = kwargs.pop("model", self._model)
        try:
            resp = await self._async_client.post(
                "/api/chat", json={"model": model, "messages": api_messages, "stream": False, **kwargs},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Ollama returned {exc.response.status_code}", provider=self.name, status_code=exc.response.status_code) from exc
        except Exception as exc:
            raise ProviderNetworkError(str(exc), provider=self.name) from exc

        data = resp.json()
        message = data.get("message", {})
        return ProviderResponse(
            text=message.get("content", ""), input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0), model=data.get("model", model),
            finish_reason=data.get("done_reason", "stop") if data.get("done") else "", raw=data,
        )

    async def async_stream(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        api_messages = self._build_messages(system_prompt, messages)
        model = kwargs.pop("model", self._model)
        try:
            async with self._async_client.stream(
                "POST", "/api/chat",
                json={"model": model, "messages": api_messages, "stream": True, **kwargs},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = _json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderNetworkError(str(exc), provider=self.name) from exc
