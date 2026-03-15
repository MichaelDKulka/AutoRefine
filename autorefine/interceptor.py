"""Invisible middleware that sits between the developer and the LLM provider.

The :class:`Interceptor` is the core of AutoRefine's "zero-code-change"
promise.  It:

1. **Looks up** the current active :class:`PromptVersion` for the given
   ``prompt_key``, silently swapping in the refined system prompt.
2. **Calls** the provider (``chat`` for non-streaming, ``stream`` for
   streaming).
3. **Logs** the full :class:`Interaction` — messages, response, tokens,
   cost, timestamps — to the store.
4. **Returns** a :class:`CompletionResponse` that carries the
   ``interaction_id`` the developer needs to submit feedback.

If *any* store operation fails (disk full, DB down, etc.), the error is
logged but **never propagated**.  The LLM call always completes — the
interceptor must be invisible and must never degrade the developer
experience.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from typing import Any

from autorefine._retry import async_retry_provider_call, retry_provider_call
from autorefine.models import (
    CompletionResponse,
    Interaction,
    Message,
    MessageRole,
)
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.interceptor")

# ---------------------------------------------------------------------------
# Design invariant: the interceptor NEVER raises an exception that would
# break the developer's app.  If anything in the AutoRefine layer fails
# (store, prompt lookup, cost estimation, logging), we log the error and
# pass the call straight through to the provider as if the SDK wasn't
# there.  Provider errors themselves DO propagate — those are the
# developer's problem, not ours.
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_interaction_id() -> str:
    """Generate a globally unique interaction ID (full uuid4 hex, 32 chars)."""
    return uuid.uuid4().hex


class Interceptor:
    """Transparent middleware that injects refined prompts and logs interactions.

    The interceptor is designed to be **invisible**: it never raises on
    store failures, never mutates the developer's messages, and returns
    the same data shape the developer would get from calling the provider
    directly — plus an ``interaction_id`` for feedback.

    Args:
        provider: The LLM provider to delegate calls to.
        store: The storage backend for reading prompt versions and
            writing interactions.
        prompt_key: The prompt namespace to use for version lookups.
        fallback_system: System prompt to use when no refined version
            exists and the caller does not supply one.

    Usage (internal — the :class:`~autorefine.client.AutoRefine` client
    creates the interceptor automatically)::

        interceptor = Interceptor(provider, store, prompt_key="support")

        # Non-streaming
        resp = interceptor.intercept_call(
            prompt_key="support",
            system_prompt="Be helpful.",
            messages_or_prompt=[{"role": "user", "content": "Hi"}],
            call_type="chat",
        )
        print(resp.text, resp.id)

        # Streaming
        for chunk in interceptor.intercept_call(
            prompt_key="support",
            system_prompt="Be helpful.",
            messages_or_prompt=[{"role": "user", "content": "Hi"}],
            call_type="stream",
        ):
            print(chunk, end="")
    """

    def __init__(
        self,
        provider: BaseProvider,
        store: BaseStore,
        prompt_key: str = "default",
        fallback_system: str = "",
    ) -> None:
        self._provider = provider
        self._store = store
        self._prompt_key = prompt_key
        self._fallback_system = fallback_system

    # ── Public API ───────────────────────────────────────────────────

    def intercept_call(
        self,
        prompt_key: str = "",
        system_prompt: str = "",
        messages_or_prompt: list[Message] | list[dict[str, str]] | str = "",
        call_type: str = "chat",
        **kwargs: Any,
    ) -> CompletionResponse | Iterator[str]:
        """Main entry point — intercepts any LLM call.

        Args:
            prompt_key: Prompt namespace for version lookup.  Falls back
                to the interceptor's default ``prompt_key``.
            system_prompt: The developer's original system prompt.  If a
                refined version exists in the store, it silently replaces
                this.  If no refined version exists, this value (or
                ``fallback_system``) is used as-is.
            messages_or_prompt: Either a list of message dicts/objects
                (for ``call_type="chat"`` or ``"stream"``) or a plain
                string (for ``call_type="complete"``).
            call_type: ``"chat"`` (default), ``"complete"`` (single
                prompt string), or ``"stream"`` (yields text chunks).
            **kwargs: Forwarded to the provider (``temperature``,
                ``max_tokens``, ``tools``, etc.).

        Returns:
            - For ``"chat"`` / ``"complete"``: a :class:`CompletionResponse`.
            - For ``"stream"``: an :class:`Iterator[str]` that yields
              text chunks and logs the interaction after the stream is
              exhausted.
        """
        key = prompt_key or self._prompt_key

        try:
            if call_type == "stream":
                messages = self._coerce_messages(messages_or_prompt)
                return self._do_stream(key, system_prompt, messages, **kwargs)

            if call_type == "complete" and isinstance(messages_or_prompt, str):
                messages = [Message(role=MessageRole.USER, content=messages_or_prompt)]
            else:
                messages = self._coerce_messages(messages_or_prompt)

            return self._do_chat(key, system_prompt, messages, **kwargs)

        except Exception as exc:
            # If the error originated from the LLM provider itself, let it
            # propagate — that's the developer's problem to handle.
            from autorefine.exceptions import ProviderError
            if isinstance(exc, ProviderError):
                raise

            # Everything else is an AutoRefine internal failure.  Log it
            # and fall through to a raw provider call so the developer's
            # app is never degraded by our SDK.
            logger.error(
                "AutoRefine internal error — falling through to raw provider "
                "call so the app is not affected: %s",
                exc, exc_info=True,
            )
            return self._fallthrough(system_prompt, messages_or_prompt, call_type, **kwargs)

    # Keep the original convenience signatures so existing callers
    # (client.py, tests) don't break.

    def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> CompletionResponse:
        """Non-streaming call with message list (backward-compatible)."""
        return self.intercept_call(
            system_prompt=system,
            messages_or_prompt=messages,
            call_type="chat",
            **kwargs,
        )

    def stream(
        self,
        messages: list[Message] | list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> Iterator[str]:
        """Streaming call with message list (backward-compatible)."""
        return self.intercept_call(
            system_prompt=system,
            messages_or_prompt=messages,
            call_type="stream",
            **kwargs,
        )

    # ── Core implementation ──────────────────────────────────────────

    def _resolve_system_prompt(
        self, prompt_key: str, developer_prompt: str
    ) -> tuple[str, int, str]:
        """Look up the active refined prompt, falling back gracefully.

        Returns:
            A 3-tuple of ``(actual_prompt_used, version, original_prompt)``.
            ``original_prompt`` is always the developer's input (for
            auditing); ``actual_prompt_used`` may differ if a refined
            version was found.
        """
        original = developer_prompt or self._fallback_system

        try:
            active = self._store.get_active_prompt(prompt_key)
        except Exception:
            logger.warning(
                "Failed to look up active prompt for key=%s — using developer prompt",
                prompt_key,
                exc_info=True,
            )
            return original, 0, original

        if active is not None:
            return active.system_prompt, active.version, original

        return original, 0, original

    def _do_chat(
        self,
        prompt_key: str,
        developer_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a non-streaming LLM call and log the interaction."""
        logger.debug(
            "Intercepting chat call: prompt_key=%s, messages=%d",
            prompt_key, len(messages),
        )
        actual_prompt, version, original_prompt = self._resolve_system_prompt(
            prompt_key, developer_prompt
        )

        # ---- Call the provider (with retry) ----
        provider_resp: ProviderResponse = retry_provider_call(
            self._provider.chat, actual_prompt, messages, **kwargs
        )

        # ---- Build & persist the Interaction ----
        interaction_id = _new_interaction_id()
        cost = self._safe_estimate_cost(
            provider_resp.input_tokens, provider_resp.output_tokens
        )

        interaction = Interaction(
            id=interaction_id,
            prompt_key=prompt_key,
            prompt_version=version,
            system_prompt=actual_prompt,
            messages=messages,
            response_text=provider_resp.text,
            input_tokens=provider_resp.input_tokens,
            output_tokens=provider_resp.output_tokens,
            model=provider_resp.model or "",
            provider=self._provider.name,
            cost_usd=cost,
            metadata={
                "original_system_prompt": original_prompt,
                "finish_reason": provider_resp.finish_reason,
            },
        )

        self._safe_save(interaction)

        # ---- Return the wrapped response ----
        return CompletionResponse(
            id=interaction_id,
            text=provider_resp.text,
            messages=messages + [
                Message(role=MessageRole.ASSISTANT, content=provider_resp.text)
            ],
            model=provider_resp.model or "",
            input_tokens=provider_resp.input_tokens,
            output_tokens=provider_resp.output_tokens,
            cost_usd=cost,
            prompt_version=version,
            metadata={
                "finish_reason": provider_resp.finish_reason,
                "tool_calls": provider_resp.tool_calls,
            },
        )

    def _do_stream(
        self,
        prompt_key: str,
        developer_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Execute a streaming LLM call, yielding chunks."""
        logger.debug(
            "Intercepting stream call: prompt_key=%s, messages=%d",
            prompt_key, len(messages),
        )
        actual_prompt, version, original_prompt = self._resolve_system_prompt(
            prompt_key, developer_prompt
        )

        interaction_id = _new_interaction_id()
        chunks: list[str] = []
        started_at = _utc_now()

        try:
            for chunk in self._provider.stream(actual_prompt, messages, **kwargs):
                chunks.append(chunk)
                yield chunk
        finally:
            # Log the interaction even if the stream is abandoned or errors.
            full_text = "".join(chunks)
            interaction = Interaction(
                id=interaction_id,
                prompt_key=prompt_key,
                prompt_version=version,
                system_prompt=actual_prompt,
                messages=messages,
                response_text=full_text,
                model="",
                provider=self._provider.name,
                created_at=started_at,
                metadata={
                    "original_system_prompt": original_prompt,
                    "streamed": True,
                },
            )
            self._safe_save(interaction)

    # ── Safety wrappers ──────────────────────────────────────────────
    # These ensure that a broken store / cost estimator never prevents
    # the LLM call from completing.

    def _safe_save(self, interaction: Interaction) -> None:
        """Save an interaction, swallowing and logging any error."""
        try:
            self._store.save_interaction(interaction)
        except Exception:
            logger.warning(
                "Failed to save interaction %s — data will be missing from "
                "analytics but the LLM call was not affected",
                interaction.id,
                exc_info=True,
            )

    def _safe_estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost, returning 0.0 if the provider raises."""
        try:
            return self._provider.estimate_cost(input_tokens, output_tokens)
        except Exception:
            logger.warning("Cost estimation failed — recording $0.00", exc_info=True)
            return 0.0

    # ── Message normalisation ────────────────────────────────────────

    @staticmethod
    def _coerce_messages(
        raw: list[Message] | list[dict[str, str]] | str,
    ) -> list[Message]:
        """Convert any supported input shape into a list of Message objects.

        Accepts:
        - ``list[Message]`` — returned as-is.
        - ``list[dict]`` — each dict converted via ``Message(role=..., content=...)``.
        - ``str`` — wrapped as a single user message.
        """
        if isinstance(raw, str):
            return [Message(role=MessageRole.USER, content=raw)]

        result: list[Message] = []
        for m in raw:
            if isinstance(m, Message):
                result.append(m)
            elif isinstance(m, dict):
                result.append(
                    Message(
                        role=MessageRole(m.get("role", "user")),
                        content=m.get("content", ""),
                    )
                )
        return result

    # ── Fallthrough — raw provider call when AutoRefine fails ─────────

    def _fallthrough(
        self,
        system_prompt: str,
        messages_or_prompt: list | str,
        call_type: str,
        **kwargs: Any,
    ) -> CompletionResponse | Iterator[str]:
        """Bypass all AutoRefine logic and call the provider directly."""
        messages = self._coerce_messages(messages_or_prompt)
        prompt = system_prompt or self._fallback_system

        if call_type == "stream":
            return self._provider.stream(prompt, messages, **kwargs)

        resp = self._provider.chat(prompt, messages, **kwargs)
        return CompletionResponse(
            id=_new_interaction_id(),
            text=resp.text,
            model=resp.model or "",
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )

    # ══════════════════════════════════════════════════════════════════
    # Async API
    # ══════════════════════════════════════════════════════════════════

    async def async_intercept_call(
        self,
        prompt_key: str = "",
        system_prompt: str = "",
        messages_or_prompt: list[Message] | list[dict[str, str]] | str = "",
        call_type: str = "chat",
        **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[str]:
        """Async version of intercept_call."""
        key = prompt_key or self._prompt_key
        try:
            if call_type == "stream":
                messages = self._coerce_messages(messages_or_prompt)
                return self._async_do_stream(key, system_prompt, messages, **kwargs)

            if call_type == "complete" and isinstance(messages_or_prompt, str):
                messages = [Message(role=MessageRole.USER, content=messages_or_prompt)]
            else:
                messages = self._coerce_messages(messages_or_prompt)

            return await self._async_do_chat(key, system_prompt, messages, **kwargs)

        except Exception as exc:
            from autorefine.exceptions import ProviderError
            if isinstance(exc, ProviderError):
                raise
            logger.error("AutoRefine async internal error — falling through: %s", exc, exc_info=True)
            return await self._async_fallthrough(system_prompt, messages_or_prompt, call_type, **kwargs)

    async def _async_do_chat(
        self, prompt_key: str, developer_prompt: str,
        messages: list[Message], **kwargs: Any,
    ) -> CompletionResponse:
        logger.debug("Async intercepting chat: prompt_key=%s", prompt_key)
        actual_prompt, version, original_prompt = self._resolve_system_prompt(prompt_key, developer_prompt)

        provider_resp = await async_retry_provider_call(
            self._provider.async_chat, actual_prompt, messages, **kwargs,
        )

        interaction_id = _new_interaction_id()
        cost = self._safe_estimate_cost(provider_resp.input_tokens, provider_resp.output_tokens)
        interaction = Interaction(
            id=interaction_id, prompt_key=prompt_key, prompt_version=version,
            system_prompt=actual_prompt, messages=messages,
            response_text=provider_resp.text, input_tokens=provider_resp.input_tokens,
            output_tokens=provider_resp.output_tokens, model=provider_resp.model or "",
            provider=self._provider.name, cost_usd=cost,
            metadata={"original_system_prompt": original_prompt, "finish_reason": provider_resp.finish_reason},
        )
        self._safe_save(interaction)

        return CompletionResponse(
            id=interaction_id, text=provider_resp.text,
            messages=messages + [Message(role=MessageRole.ASSISTANT, content=provider_resp.text)],
            model=provider_resp.model or "", input_tokens=provider_resp.input_tokens,
            output_tokens=provider_resp.output_tokens, cost_usd=cost, prompt_version=version,
            metadata={"finish_reason": provider_resp.finish_reason, "tool_calls": provider_resp.tool_calls},
        )

    async def _async_do_stream(
        self, prompt_key: str, developer_prompt: str,
        messages: list[Message], **kwargs: Any,
    ) -> AsyncIterator[str]:
        logger.debug("Async intercepting stream: prompt_key=%s", prompt_key)
        actual_prompt, version, original_prompt = self._resolve_system_prompt(prompt_key, developer_prompt)
        interaction_id = _new_interaction_id()
        chunks: list[str] = []
        started_at = _utc_now()
        try:
            async for chunk in self._provider.async_stream(actual_prompt, messages, **kwargs):
                chunks.append(chunk)
                yield chunk
        finally:
            full_text = "".join(chunks)
            interaction = Interaction(
                id=interaction_id, prompt_key=prompt_key, prompt_version=version,
                system_prompt=actual_prompt, messages=messages, response_text=full_text,
                model="", provider=self._provider.name, created_at=started_at,
                metadata={"original_system_prompt": original_prompt, "streamed": True},
            )
            self._safe_save(interaction)

    async def _async_fallthrough(
        self, system_prompt: str, messages_or_prompt: list | str,
        call_type: str, **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[str]:
        messages = self._coerce_messages(messages_or_prompt)
        prompt = system_prompt or self._fallback_system
        if call_type == "stream":
            return self._provider.async_stream(prompt, messages, **kwargs)
        resp = await self._provider.async_chat(prompt, messages, **kwargs)
        return CompletionResponse(
            id=_new_interaction_id(), text=resp.text, model=resp.model or "",
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
        )
