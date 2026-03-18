"""API key validation, organization lookup, and rate limiting.

Validates incoming requests by hashing the Bearer token and looking
it up in the database. Includes an in-memory LRU cache with TTL to
avoid DB lookups on every request, and a sliding-window rate limiter.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, defaultdict
from typing import Any

from autorefine.cloud.keys import hash_key
from autorefine.cloud.models import ApiKey, Organization
from autorefine.exceptions import CloudAuthError, ProviderRateLimitError, SpendCapExceeded

logger = logging.getLogger("autorefine.cloud.auth")


class _KeyCache:
    """In-memory LRU cache with TTL for API key lookups."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 60.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Organization, ApiKey]] = OrderedDict()

    def get(self, key_hash: str) -> tuple[Organization, ApiKey] | None:
        if key_hash not in self._cache:
            return None
        ts, org, api_key = self._cache[key_hash]
        if time.monotonic() - ts > self._ttl:
            del self._cache[key_hash]
            return None
        self._cache.move_to_end(key_hash)
        return org, api_key

    def put(self, key_hash: str, org: Organization, api_key: ApiKey) -> None:
        self._cache[key_hash] = (time.monotonic(), org, api_key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


class _RateLimiter:
    """Sliding-window rate limiter per API key."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(self, key_id: str, limit_rpm: int) -> bool:
        """Return True if the request is within rate limits."""
        now = time.monotonic()
        window = self._windows[key_id]
        # Prune entries older than 60 seconds
        self._windows[key_id] = [t for t in window if now - t < 60.0]
        if len(self._windows[key_id]) >= limit_rpm:
            return False
        self._windows[key_id].append(now)
        return True


class _DailyCounter:
    """Tracks daily call count for test keys (100/day limit)."""

    def __init__(self) -> None:
        self._counts: dict[str, tuple[str, int]] = {}

    def check_and_increment(self, key_id: str, limit: int = 100) -> bool:
        today = time.strftime("%Y-%m-%d")
        day, count = self._counts.get(key_id, ("", 0))
        if day != today:
            self._counts[key_id] = (today, 1)
            return True
        if count >= limit:
            return False
        self._counts[key_id] = (today, count + 1)
        return True


class Authenticator:
    """Validates API keys and enforces rate/spend limits.

    Args:
        store: A callable that takes a key hash and returns
            ``(Organization, ApiKey)`` or ``None``. Typically backed
            by the cloud Postgres store.
        get_monthly_spend: A callable that takes an org_id and returns
            the current month's total customer spend.
    """

    def __init__(
        self,
        store: Any = None,
        get_monthly_spend: Any = None,
    ) -> None:
        self._store = store
        self._get_monthly_spend = get_monthly_spend
        self._cache = _KeyCache()
        self._limiter = _RateLimiter()
        self._daily = _DailyCounter()

    def validate(
        self, raw_key: str, model: str = "",
    ) -> tuple[Organization, ApiKey]:
        """Validate an API key and return the org + key record.

        Raises:
            CloudAuthError: Key is invalid, revoked, or org is inactive.
            SpendCapExceeded: Monthly spend cap exceeded.
            ProviderRateLimitError: Rate limit exceeded.
        """
        if not raw_key:
            raise CloudAuthError("No API key provided")

        kh = hash_key(raw_key)

        # Cache check
        cached = self._cache.get(kh)
        if cached:
            org, api_key = cached
        else:
            # DB lookup
            if self._store is None:
                raise CloudAuthError("Auth store not configured")
            result = self._store(kh)
            if result is None:
                raise CloudAuthError("Invalid AutoRefine API key")
            org, api_key = result
            self._cache.put(kh, org, api_key)

        # Active checks
        if not api_key.is_active:
            raise CloudAuthError("API key has been revoked")
        if not org.is_active:
            raise CloudAuthError("Organization is inactive")

        # Model restriction
        if model and api_key.model_restrictions and model not in api_key.model_restrictions:
            raise CloudAuthError(
                f"Model '{model}' is not allowed for this API key. "
                f"Allowed: {api_key.model_restrictions}"
            )

        # Rate limit
        if not self._limiter.check(api_key.id, api_key.rate_limit_rpm):
            raise ProviderRateLimitError(
                f"Rate limit exceeded ({api_key.rate_limit_rpm} rpm)",
                provider="cloud",
            )

        # Test key daily limit
        if raw_key.startswith("ar_test_"):
            if not self._daily.check_and_increment(api_key.id, limit=100):
                raise ProviderRateLimitError(
                    "Test key daily limit exceeded (100 calls/day)",
                    provider="cloud",
                )

        # Spend cap check
        if self._get_monthly_spend:
            cap = api_key.monthly_spend_cap or org.monthly_spend_cap
            current = self._get_monthly_spend(org.id)
            if current >= cap:
                raise SpendCapExceeded(
                    f"Monthly spend cap ${cap:.2f} exceeded (current: ${current:.2f})",
                    current_spend=current,
                    cap=cap,
                )

        # Update last_used_at
        api_key.last_used_at = None  # Will be set by the caller on save

        return org, api_key
