"""Abstract base class for LLM providers.

Every provider implements three sync call patterns:

- ``complete(system_prompt, user_prompt)`` — one-shot convenience
- ``chat(system_prompt, messages)`` — multi-turn conversation
- ``stream(system_prompt, messages)`` — streaming multi-turn

Async variants are provided with ``asyncio.to_thread`` fallback.
Providers that have native async SDKs (OpenAI, Anthropic) override
the async methods directly for better performance.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from autorefine.models import Message, MessageRole


@dataclass
class ProviderResponse:
    """Standardised response from any LLM provider."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    tool_calls: list[Any] = field(default_factory=list)
    raw: Any = None


class BaseProvider(ABC):
    """Interface every LLM provider must implement.

    Subclasses must override :meth:`chat`, :meth:`stream`, and
    :meth:`estimate_cost`.  Async methods have default implementations
    that delegate to sync via ``asyncio.to_thread``; providers with
    native async SDKs should override them for zero-overhead async.
    """

    name: str = "base"

    # ── Sync abstract methods ────────────────────────────────────────

    @abstractmethod
    def chat(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> ProviderResponse:
        """Send a multi-turn chat completion (sync)."""
        ...

    @abstractmethod
    def stream(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream a chat completion, yielding text chunks (sync)."""
        ...

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return the estimated cost in USD for the given token counts."""
        ...

    # ── Sync convenience ─────────────────────────────────────────────

    def complete(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> ProviderResponse:
        """One-shot convenience: wraps chat() with a single user message."""
        return self.chat(system_prompt, [Message(role=MessageRole.USER, content=user_prompt)], **kwargs)

    # ── Async methods (default: delegate to sync via to_thread) ──────

    async def async_chat(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> ProviderResponse:
        """Async multi-turn chat. Override for native async performance."""
        return await asyncio.to_thread(self.chat, system_prompt, messages, **kwargs)

    async def async_complete(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> ProviderResponse:
        """Async one-shot convenience."""
        return await asyncio.to_thread(self.complete, system_prompt, user_prompt, **kwargs)

    async def async_stream(self, system_prompt: str, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        """Async streaming. Override for native async streaming."""
        # Default: run sync stream in a thread, yield results via a queue
        import queue
        q: queue.Queue[str | None] = queue.Queue()

        def _run():
            try:
                for chunk in self.stream(system_prompt, messages, **kwargs):
                    q.put(chunk)
            finally:
                q.put(None)  # sentinel

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run)
        while True:
            chunk = await asyncio.to_thread(q.get)
            if chunk is None:
                break
            yield chunk

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_dicts(messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to plain dicts for provider SDKs."""
        return [{"role": m.role.value, "content": m.content} for m in messages]
