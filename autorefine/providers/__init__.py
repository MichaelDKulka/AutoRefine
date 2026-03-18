"""LLM provider implementations.

Use :func:`get_provider` to instantiate a provider by name or let it
auto-detect the right one from a model identifier string.

Example::

    from autorefine.providers import get_provider

    # Explicit provider name
    p = get_provider("openai", api_key="sk-...", model="gpt-4o")

    # Auto-detect from model name
    p = get_provider(model="claude-sonnet-4-20250514", api_key="sk-ant-...")
    p = get_provider(model="llama3.1")  # Ollama — no key needed
"""

from __future__ import annotations

from typing import Any

from autorefine.providers.base import BaseProvider, ProviderResponse

# Model-name substrings → provider name
_MODEL_HINTS: list[tuple[list[str], str]] = [
    (["gpt", "o1", "o3", "o4", "davinci", "chatgpt"], "openai"),
    (["claude"], "anthropic"),
    (["mistral", "mixtral", "codestral"], "mistral"),
    (["llama", "gemma", "phi", "qwen", "deepseek", "codellama", "command-r"], "ollama"),
]


def _detect_provider(model: str) -> str:
    """Guess the provider name from a model identifier string."""
    lower = model.lower()
    for keywords, provider_name in _MODEL_HINTS:
        if any(kw in lower for kw in keywords):
            return provider_name
    return "openai"  # safe default


def get_provider(
    provider_name: str = "",
    *,
    api_key: str = "",
    model: str = "",
    **kwargs: Any,
) -> BaseProvider:
    """Factory function that returns a configured provider instance.

    Args:
        provider_name: Explicit provider (``"openai"``, ``"anthropic"``,
            ``"mistral"``, ``"ollama"``, ``"cloud"``).  When empty, the
            provider is auto-detected from *model* or *api_key*.
        api_key: API key for the provider.  Not needed for Ollama.
            Keys starting with ``ar_live_`` or ``ar_test_`` automatically
            route through AutoRefine Cloud.
        model: Model identifier.  Used for auto-detection when
            *provider_name* is empty, and forwarded to the provider.
        **kwargs: Extra arguments forwarded to the provider constructor
            (e.g. ``base_url``, ``max_tokens``, ``timeout``).

    Returns:
        A configured :class:`BaseProvider` instance ready for use.

    Raises:
        ValueError: If the provider name is unrecognised.
        ImportError: If the required SDK package is not installed.
    """
    # Detect AutoRefine Cloud keys before any other routing
    if api_key and (api_key.startswith("ar_live_") or api_key.startswith("ar_test_")):
        from autorefine.providers.cloud_provider import CloudProvider

        return CloudProvider(api_key=api_key, model=model or "gpt-4o", **kwargs)

    name = provider_name or _detect_provider(model)

    if name == "openai":
        from autorefine.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o", **kwargs)

    if name == "anthropic":
        from autorefine.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
            **kwargs,
        )

    if name == "mistral":
        from autorefine.providers.mistral_provider import MistralProvider

        return MistralProvider(
            api_key=api_key,
            model=model or "mistral-small-latest",
            **kwargs,
        )

    if name == "ollama":
        from autorefine.providers.ollama_provider import OllamaProvider

        return OllamaProvider(model=model or "llama3", **kwargs)

    raise ValueError(
        f"Unknown provider '{name}'. "
        f"Supported: openai, anthropic, mistral, ollama"
    )


__all__ = ["BaseProvider", "ProviderResponse", "get_provider"]
