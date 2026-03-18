"""LLM proxy -- forwards requests to upstream providers and applies markup.

The proxy is the core of the cloud server. It:
1. Resolves the upstream provider from the model name
2. Gets the upstream API key (BYOK or Upwell master key)
3. Applies the refined prompt if one exists
4. Forwards the call to the upstream provider
5. Calculates and records cost with markup
6. Returns the response + cost metadata
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from autorefine.cloud.billing import BillingManager, calculate_cost
from autorefine.cloud.models import ApiKey, Organization
from autorefine.models import Interaction, Message, MessageRole
from autorefine.providers import _detect_provider, get_provider
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.cloud.proxy")

# Upwell master keys for upstream providers (used when org has no BYOK)
_MASTER_KEYS: dict[str, str] = {}


def _get_master_key(provider_name: str) -> str:
    """Get the Upwell master API key for an upstream provider."""
    if not _MASTER_KEYS:
        _MASTER_KEYS["openai"] = os.environ.get("UPWELL_OPENAI_KEY", "")
        _MASTER_KEYS["anthropic"] = os.environ.get("UPWELL_ANTHROPIC_KEY", "")
        _MASTER_KEYS["mistral"] = os.environ.get("UPWELL_MISTRAL_KEY", "")
    return _MASTER_KEYS.get(provider_name, "")


@dataclass
class ProxyResponse:
    """Response from a proxied LLM call with cost metadata."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    interaction_id: str = ""
    upstream_cost: float = 0.0
    markup_amount: float = 0.0
    customer_cost: float = 0.0


class LLMProxy:
    """Forwards LLM requests to upstream providers through the cloud.

    Args:
        store: Storage backend for interactions and prompts.
        billing: Billing manager for usage metering.
    """

    def __init__(
        self,
        store: BaseStore,
        billing: BillingManager,
    ) -> None:
        self._store = store
        self._billing = billing
        self._provider_cache: dict[str, BaseProvider] = {}

    def _get_upstream_provider(
        self,
        model: str,
        org: Organization,
    ) -> BaseProvider:
        """Resolve and cache the upstream provider for a model."""
        provider_name = _detect_provider(model)

        # Get API key: BYOK first, then Upwell master key
        api_key = org.upstream_keys.get(provider_name, "")
        if not api_key:
            api_key = _get_master_key(provider_name)

        if not api_key:
            raise ValueError(
                f"No API key available for provider '{provider_name}'. "
                f"Configure BYOK or set UPWELL_{provider_name.upper()}_KEY."
            )

        cache_key = f"{provider_name}:{api_key[:8]}:{model}"
        if cache_key not in self._provider_cache:
            self._provider_cache[cache_key] = get_provider(
                provider_name, api_key=api_key, model=model,
            )
        return self._provider_cache[cache_key]

    def _get_refined_prompt(
        self,
        system_prompt: str,
        prompt_key: str,
        namespace: str = "",
    ) -> str:
        """Look up the active refined prompt, falling back to the original."""
        try:
            ns_key = f"{namespace}:{prompt_key}" if namespace else prompt_key
            active = self._store.get_active_prompt(ns_key)
            if active:
                return active.system_prompt
        except Exception:
            logger.debug("Failed to look up refined prompt for %s", prompt_key)
        return system_prompt

    def chat(
        self,
        org: Organization,
        api_key: ApiKey,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        prompt_key: str = "default",
        **kwargs: Any,
    ) -> ProxyResponse:
        """Proxy a chat completion to the upstream provider.

        All store/billing writes are fire-and-forget — never block the
        LLM response.
        """
        provider = self._get_upstream_provider(model, org)

        # Apply refined prompt if available
        namespace = org.slug or org.id
        effective_prompt = self._get_refined_prompt(system_prompt, prompt_key, namespace)

        # Convert message dicts to Message objects
        msg_objects = [
            Message(role=MessageRole(m.get("role", "user")), content=m.get("content", ""))
            for m in messages
        ]

        # Forward to upstream
        resp = provider.chat(effective_prompt, msg_objects, **kwargs)

        # Calculate cost
        upstream, markup, customer = calculate_cost(
            org, model, resp.input_tokens, resp.output_tokens,
        )

        # Log interaction (fire-and-forget)
        ix = Interaction(
            prompt_key=f"{namespace}:{prompt_key}" if namespace else prompt_key,
            system_prompt=effective_prompt,
            messages=msg_objects,
            response_text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            model=resp.model or model,
            provider=provider.name,
            cost_usd=customer,
        )
        try:
            self._store.save_interaction(ix)
        except Exception:
            logger.warning("Failed to save interaction %s", ix.id[:8], exc_info=True)

        # Record usage (fire-and-forget)
        try:
            self._billing.record_usage(
                org=org,
                api_key_id=api_key.id,
                model=resp.model or model,
                provider=provider.name,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                interaction_id=ix.id,
                prompt_key=prompt_key,
            )
        except Exception:
            logger.warning("Failed to record usage", exc_info=True)

        return ProxyResponse(
            text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            model=resp.model or model,
            finish_reason=resp.finish_reason,
            interaction_id=ix.id,
            upstream_cost=upstream,
            markup_amount=markup,
            customer_cost=customer,
        )

    def stream(
        self,
        org: Organization,
        api_key: ApiKey,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        prompt_key: str = "default",
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Proxy a streaming chat completion.

        Yields NDJSON-compatible dicts: ``{"text": "chunk"}`` for each
        chunk, and ``{"done": true, "usage": {...}}`` at the end.
        """
        provider = self._get_upstream_provider(model, org)

        namespace = org.slug or org.id
        effective_prompt = self._get_refined_prompt(system_prompt, prompt_key, namespace)

        msg_objects = [
            Message(role=MessageRole(m.get("role", "user")), content=m.get("content", ""))
            for m in messages
        ]

        full_text = []
        for chunk in provider.stream(effective_prompt, msg_objects, **kwargs):
            full_text.append(chunk)
            yield {"text": chunk}

        # After stream completes, log asynchronously
        response_text = "".join(full_text)

        # Estimate tokens (approximate for streaming)
        est_input = len(system_prompt.split()) + sum(len(m.get("content", "").split()) for m in messages)
        est_output = len(response_text.split())
        # Rough token estimate: ~0.75 tokens per word
        est_input_tokens = int(est_input * 1.3)
        est_output_tokens = int(est_output * 1.3)

        upstream, markup, customer = calculate_cost(
            org, model, est_input_tokens, est_output_tokens,
        )

        # Log interaction
        ix = Interaction(
            prompt_key=f"{namespace}:{prompt_key}" if namespace else prompt_key,
            system_prompt=effective_prompt,
            messages=msg_objects,
            response_text=response_text,
            input_tokens=est_input_tokens,
            output_tokens=est_output_tokens,
            model=model,
            provider=provider.name,
            cost_usd=customer,
        )
        try:
            self._store.save_interaction(ix)
        except Exception:
            logger.warning("Failed to save streamed interaction", exc_info=True)

        try:
            self._billing.record_usage(
                org=org,
                api_key_id=api_key.id,
                model=model,
                provider=provider.name,
                input_tokens=est_input_tokens,
                output_tokens=est_output_tokens,
                interaction_id=ix.id,
                prompt_key=prompt_key,
            )
        except Exception:
            logger.warning("Failed to record streaming usage", exc_info=True)

        yield {
            "done": True,
            "usage": {
                "interaction_id": ix.id,
                "input_tokens": est_input_tokens,
                "output_tokens": est_output_tokens,
                "customer_cost": customer,
            },
        }
