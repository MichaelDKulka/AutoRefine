"""AutoRefine Cloud proxy provider.

Routes LLM calls through the AutoRefine Cloud API instead of directly
to upstream providers. Activated automatically when the api_key starts
with 'ar_live_' or 'ar_test_'.

The cloud proxy:
1. Authenticates the request using the AutoRefine API key
2. Forwards to the upstream LLM provider (OpenAI, Anthropic, etc.)
3. Logs the interaction for analytics and refinement
4. Applies the refined prompt if one exists
5. Returns the response identically to a direct provider call

Usage::

    from autorefine.providers.cloud_provider import CloudProvider

    provider = CloudProvider(api_key="ar_live_abc123...", model="gpt-4o")
    resp = provider.complete("You are helpful.", "What is 2+2?")
"""

from __future__ import annotations

import json as _json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from autorefine.exceptions import (
    CloudAuthError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderResponseError,
    SpendCapExceeded,
)
from autorefine.models import Message
from autorefine.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger("autorefine.provider.cloud")


def _classify_error(status_code: int, body: str) -> Exception:
    """Map HTTP status codes to the AutoRefine exception hierarchy."""
    if status_code in (401, 403):
        return CloudAuthError(
            f"Invalid AutoRefine API key (HTTP {status_code}): {body[:200]}"
        )
    if status_code == 402:
        return SpendCapExceeded(
            f"Spend cap exceeded: {body[:200]}"
        )
    if status_code == 429:
        return ProviderRateLimitError(
            f"Rate limit exceeded: {body[:200]}",
            provider="cloud",
        )
    if status_code >= 500:
        return ProviderNetworkError(
            f"AutoRefine Cloud server error (HTTP {status_code}): {body[:200]}",
            provider="cloud",
        )
    return ProviderResponseError(
        f"AutoRefine Cloud error (HTTP {status_code}): {body[:200]}",
        provider="cloud",
    )


class CloudProvider(BaseProvider):
    """HTTP proxy provider that routes calls through AutoRefine Cloud.

    Args:
        api_key: AutoRefine API key (``ar_live_...`` or ``ar_test_...``).
        model: Default model identifier (e.g. ``"gpt-4o"``).
        base_url: Cloud API base URL. Defaults to ``https://api.autorefine.dev``.
        timeout: Request timeout in seconds.
        **kwargs: Extra arguments forwarded to ``httpx.Client()``.

    Usage::

        provider = CloudProvider(api_key="ar_live_abc...", model="gpt-4o")
        resp = provider.chat("Be helpful.", messages)
    """

    name = "cloud"

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        base_url: str = "https://api.autorefine.dev",
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "autorefine-sdk/0.2.0",
        }
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=self._headers,
            **kwargs,
        )
        self._async_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=self._headers,
            **kwargs,
        )

    # ── Sync API ──────────────────────────────────────────────────────

    def chat(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a chat request through the cloud proxy."""
        model = kwargs.pop("model", self._model)
        prompt_key = kwargs.pop("prompt_key", "default")

        body = {
            "model": model,
            "system_prompt": system_prompt,
            "messages": self._to_dicts(messages),
            "prompt_key": prompt_key,
            **kwargs,
        }

        try:
            resp = self._client.post("/v1/chat", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderNetworkError(
                f"AutoRefine Cloud request timed out after {self._timeout}s",
                provider="cloud",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderNetworkError(
                f"Cannot connect to AutoRefine Cloud at {self._base_url}: {exc}",
                provider="cloud",
            ) from exc

        if resp.status_code != 200:
            raise _classify_error(resp.status_code, resp.text)

        data = resp.json()
        return ProviderResponse(
            text=data.get("text", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            model=data.get("model", model),
            finish_reason=data.get("finish_reason", ""),
            raw=data,
        )

    def stream(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a chat completion through the cloud proxy (NDJSON)."""
        model = kwargs.pop("model", self._model)
        prompt_key = kwargs.pop("prompt_key", "default")

        body = {
            "model": model,
            "system_prompt": system_prompt,
            "messages": self._to_dicts(messages),
            "prompt_key": prompt_key,
            **kwargs,
        }

        try:
            with self._client.stream("POST", "/v1/chat/stream", json=body) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise _classify_error(resp.status_code, resp.text)

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        break
                    text = chunk.get("text", "")
                    if text:
                        yield text
        except httpx.TimeoutException as exc:
            raise ProviderNetworkError(
                f"Cloud stream timed out after {self._timeout}s",
                provider="cloud",
            ) from exc

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost using the local pricing table.

        The actual cost (with markup) is calculated server-side and
        returned in the response metadata. This is a best-effort
        client-side estimate for budget checking.
        """
        from autorefine.cost_tracker import estimate_cost
        return estimate_cost(self._model, input_tokens, output_tokens)

    # ── Async API ─────────────────────────────────────────────────────

    async def async_chat(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Native async chat through the cloud proxy."""
        model = kwargs.pop("model", self._model)
        prompt_key = kwargs.pop("prompt_key", "default")

        body = {
            "model": model,
            "system_prompt": system_prompt,
            "messages": self._to_dicts(messages),
            "prompt_key": prompt_key,
            **kwargs,
        }

        try:
            resp = await self._async_client.post("/v1/chat", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderNetworkError(
                f"AutoRefine Cloud request timed out after {self._timeout}s",
                provider="cloud",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderNetworkError(
                f"Cannot connect to AutoRefine Cloud at {self._base_url}: {exc}",
                provider="cloud",
            ) from exc

        if resp.status_code != 200:
            raise _classify_error(resp.status_code, resp.text)

        data = resp.json()
        return ProviderResponse(
            text=data.get("text", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            model=data.get("model", model),
            finish_reason=data.get("finish_reason", ""),
            raw=data,
        )

    async def async_stream(
        self,
        system_prompt: str,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Native async streaming through the cloud proxy."""
        model = kwargs.pop("model", self._model)
        prompt_key = kwargs.pop("prompt_key", "default")

        body = {
            "model": model,
            "system_prompt": system_prompt,
            "messages": self._to_dicts(messages),
            "prompt_key": prompt_key,
            **kwargs,
        }

        try:
            async with self._async_client.stream(
                "POST", "/v1/chat/stream", json=body
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise _classify_error(resp.status_code, resp.text)

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        break
                    text = chunk.get("text", "")
                    if text:
                        yield text
        except httpx.TimeoutException as exc:
            raise ProviderNetworkError(
                f"Cloud stream timed out after {self._timeout}s",
                provider="cloud",
            ) from exc

    # ── Cloud-specific: feedback submission ────────────────────────────

    def submit_feedback(
        self,
        interaction_id: str,
        signal: str,
        comment: str = "",
        dimensions: dict[str, float] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit feedback through the cloud API."""
        body: dict[str, Any] = {
            "interaction_id": interaction_id,
            "signal": signal,
        }
        if comment:
            body["comment"] = comment
        if dimensions:
            body["dimensions"] = dimensions
        if context:
            body["context"] = context

        try:
            resp = self._client.post("/v1/feedback", json=body)
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.warning("Cloud feedback submission failed: %s", exc)
            return {"status": "error", "message": str(exc)}

        if resp.status_code != 200:
            logger.warning("Cloud feedback returned %d", resp.status_code)
            return {"status": "error", "code": resp.status_code}

        return resp.json()
