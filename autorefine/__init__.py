"""AutoRefine — make any AI app progressively smarter with automatic prompt refinement.

Quick start::

    from autorefine import AutoRefine

    client = AutoRefine(api_key="sk-...", model="gpt-4o",
                        refiner_key="sk-ant-...", auto_learn=True)
    resp = client.complete("You are helpful.", "What is 2+2?")
    client.feedback(resp.id, "thumbs_up")

Decorator usage::

    from autorefine import autorefine

    @autorefine(api_key="sk-...", refiner_key="sk-ant-...", auto_learn=True)
    def ask(system: str, prompt: str):
        # your existing LLM call logic — AutoRefine wraps it transparently
        return {"system": system, "prompt": prompt}
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from autorefine.__version__ import __version__
from autorefine.async_client import AsyncAutoRefine
from autorefine.client import AutoRefine
from autorefine.feedback import FeedbackBundle, FeedbackCollector
from autorefine.feedback_filter import FeedbackFilter
from autorefine.feedback_provider import FeedbackProvider
from autorefine.models import (
    CompletionResponse,
    FeedbackSignal,
    FeedbackType,
    Message,
    PromptVersion,
)
from autorefine.notifications import PromptChangeEvent, PromptChangeNotifier
from autorefine.pii_scrubber import PIIScrubber
from autorefine.widget import FeedbackWidget

logger = logging.getLogger("autorefine")

# ── Startup banner (shown once on first import, like Claude Code) ────

_BANNER = r"""
    _         _        ____       __ _
   / \  _   _| |_ ___ |  _ \ ___ / _(_)_ __   ___
  / _ \| | | | __/ _ \| |_) / _ \ |_| | '_ \ / _ \
 / ___ \ |_| | || (_) |  _ <  __/  _| | | | |  __/
/_/   \_\__,_|\__\___/|_| \_\___|_| |_|_| |_|\___|
"""


def _show_banner() -> None:
    """Print the startup banner to stderr (non-intrusive)."""
    import os
    import sys
    if os.environ.get("AUTOREFINE_QUIET") or not sys.stderr.isatty():
        return
    sys.stderr.write(f"\033[34m{_BANNER}\033[0m")
    sys.stderr.write(f"  \033[90mv{__version__} by Upwell Digital Solutions\033[0m\n\n")


_show_banner()

F = TypeVar("F", bound=Callable[..., Any])


def autorefine(
    api_key: str = "",
    model: str = "gpt-4o",
    refiner_key: str = "",
    auto_learn: bool = False,
    prompt_key: str = "default",
    **config_overrides: Any,
) -> Callable[[F], F]:
    """Decorator that wraps any function making LLM calls with AutoRefine.

    The decorated function should accept ``system`` and ``prompt`` (or
    ``messages``) as its first arguments.  AutoRefine intercepts the call,
    injects the refined system prompt if one exists, logs the interaction,
    and returns a :class:`CompletionResponse` with a ``.id`` the caller
    can use for feedback.

    Usage::

        @autorefine(api_key="sk-...", refiner_key="sk-ant-...", auto_learn=True)
        def ask(system: str, prompt: str) -> CompletionResponse:
            # The decorator intercepts — your original function body
            # is used as a fallback only if AutoRefine isn't configured.
            return {"system": system, "prompt": prompt}

        resp = ask("You are helpful.", "What is 2+2?")
        print(resp.text)
        ask.feedback(resp.id, "thumbs_up")

    The decorated function gains extra attributes:

    - ``ask.client`` — the underlying :class:`AutoRefine` instance.
    - ``ask.feedback(response_id, signal, comment=None)`` — record feedback.
    - ``ask.get_active_prompt()`` — get the current active prompt.
    - ``ask.analytics`` — access analytics.
    """
    def decorator(fn: F) -> F:
        # Lazy-init: create the client on first call
        _client: list[AutoRefine | None] = [None]

        def _get_client() -> AutoRefine:
            if _client[0] is None:
                _client[0] = AutoRefine(
                    api_key=api_key, model=model, refiner_key=refiner_key,
                    auto_learn=auto_learn, prompt_key=prompt_key,
                    **config_overrides,
                )
            return _client[0]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> CompletionResponse:
            client = _get_client()
            # Extract system and prompt from args/kwargs
            system = ""
            prompt = ""
            if args:
                system = str(args[0]) if len(args) > 0 else ""
                prompt = str(args[1]) if len(args) > 1 else ""
            system = kwargs.pop("system", system)
            prompt = kwargs.pop("prompt", prompt)
            messages = kwargs.pop("messages", None)

            if messages is not None:
                return client.chat(system, messages, prompt_key=prompt_key, **kwargs)
            elif prompt:
                return client.complete(system, prompt, prompt_key=prompt_key, **kwargs)
            else:
                # Fall back to the original function
                return fn(*args, **kwargs)

        # Attach helpers to the wrapper
        wrapper.client = property(lambda self: _get_client())  # type: ignore[attr-defined]
        wrapper.feedback = lambda resp_id, signal, **kw: _get_client().feedback(resp_id, signal, **kw)  # type: ignore[attr-defined]
        wrapper.get_active_prompt = lambda pk="": _get_client().get_active_prompt(pk)  # type: ignore[attr-defined]
        wrapper.analytics = property(lambda self: _get_client().analytics)  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    "AsyncAutoRefine",
    "AutoRefine",
    "CompletionResponse",
    "FeedbackBundle",
    "FeedbackCollector",
    "FeedbackProvider",
    "FeedbackWidget",
    "FeedbackFilter",
    "FeedbackSignal",
    "FeedbackType",
    "Message",
    "PIIScrubber",
    "PromptChangeEvent",
    "PromptChangeNotifier",
    "PromptVersion",
    "__version__",
    "autorefine",
]
