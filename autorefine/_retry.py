"""Exponential backoff retry logic for provider and store operations.

Used internally by providers, the interceptor, and store backends.
Not part of the public API.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from typing import Any, Callable, TypeVar

from autorefine.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
    StorageError,
)

logger = logging.getLogger("autorefine.retry")

T = TypeVar("T")

_DEFAULT_DELAYS = (1.0, 2.0, 4.0)  # seconds between retries


def retry_provider_call(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    delays: tuple[float, ...] = _DEFAULT_DELAYS,
    **kwargs: Any,
) -> T:
    """Call *fn* with exponential backoff retry on retryable provider errors.

    Auth errors are never retried.  Rate-limit and network errors are
    retried up to *max_attempts* times with delays of 1s, 2s, 4s.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except ProviderAuthError:
            raise  # never retry auth failures
        except (ProviderRateLimitError, ProviderNetworkError) as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                logger.warning(
                    "Provider call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_attempts, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Provider call failed after %d attempts: %s",
                    max_attempts, exc,
                )
        except ProviderError:
            raise  # non-retryable provider errors (malformed response, etc.)
        except Exception as exc:
            # Unknown exception — wrap and raise, don't retry
            raise ProviderError(
                f"Unexpected error in provider call: {exc}",
                retryable=False,
            ) from exc

    raise last_exc  # type: ignore[misc]


def retry_store_write(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    delays: tuple[float, ...] = _DEFAULT_DELAYS,
    operation: str = "store write",
    **kwargs: Any,
) -> T:
    """Call *fn* with exponential backoff retry on store write failures.

    All exceptions except ``StorageError`` are wrapped.  Returns the
    function's return value on success.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                logger.warning(
                    "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                    operation, attempt + 1, max_attempts, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation, max_attempts, exc,
                )

    raise StorageError(
        f"{operation} failed after {max_attempts} attempts: {last_exc}",
    ) from last_exc


# ── Async variants ───────────────────────────────────────────────────

async def async_retry_provider_call(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_attempts: int = 3,
    delays: tuple[float, ...] = _DEFAULT_DELAYS,
    **kwargs: Any,
) -> T:
    """Async version of retry_provider_call."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn(*args, **kwargs)
        except ProviderAuthError:
            raise
        except (ProviderRateLimitError, ProviderNetworkError) as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                logger.warning(
                    "Async provider call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_attempts, delay, exc,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Async provider call failed after %d attempts: %s", max_attempts, exc)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Unexpected error in async provider call: {exc}", retryable=False) from exc
    raise last_exc  # type: ignore[misc]
