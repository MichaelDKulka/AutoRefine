"""Tests for cloud authentication and rate limiting."""

from __future__ import annotations

import pytest

from autorefine.cloud.auth import Authenticator, _DailyCounter, _KeyCache, _RateLimiter
from autorefine.cloud.keys import hash_key
from autorefine.cloud.models import ApiKey, Organization
from autorefine.exceptions import CloudAuthError, ProviderRateLimitError, SpendCapExceeded


def _make_org(**kw):
    return Organization(id="org-1", name="Test", slug="test", **kw)


def _make_key(org_id="org-1", **kw):
    return ApiKey(org_id=org_id, key_hash=hash_key("ar_live_testkey123"), key_prefix="ar_live_testkey1", **kw)


def _store_fn(org, key):
    """Return a store callable that returns the given org/key for any hash."""
    def lookup(kh):
        if kh == key.key_hash:
            return org, key
        return None
    return lookup


# ── Basic validation ──────────────────────────────────────────────────

def test_valid_key_returns_org():
    org = _make_org()
    key = _make_key()
    auth = Authenticator(store=_store_fn(org, key))
    result_org, result_key = auth.validate("ar_live_testkey123")
    assert result_org.id == "org-1"
    assert result_key.org_id == "org-1"


def test_invalid_key_raises():
    org = _make_org()
    key = _make_key()
    auth = Authenticator(store=_store_fn(org, key))
    with pytest.raises(CloudAuthError):
        auth.validate("ar_live_wrongkey999")


def test_revoked_key_raises():
    org = _make_org()
    key = _make_key(is_active=False)
    auth = Authenticator(store=_store_fn(org, key))
    with pytest.raises(CloudAuthError, match="revoked"):
        auth.validate("ar_live_testkey123")


def test_inactive_org_raises():
    org = _make_org(is_active=False)
    key = _make_key()
    auth = Authenticator(store=_store_fn(org, key))
    with pytest.raises(CloudAuthError, match="inactive"):
        auth.validate("ar_live_testkey123")


def test_model_restriction_enforced():
    org = _make_org()
    key = _make_key(model_restrictions=["gpt-4o"])
    auth = Authenticator(store=_store_fn(org, key))
    # Allowed model
    auth.validate("ar_live_testkey123", model="gpt-4o")
    # Blocked model
    with pytest.raises(CloudAuthError, match="not allowed"):
        auth.validate("ar_live_testkey123", model="claude-sonnet-4-20250514")


def test_spend_cap_enforced():
    org = _make_org(monthly_spend_cap=100.0)
    key = _make_key()
    auth = Authenticator(
        store=_store_fn(org, key),
        get_monthly_spend=lambda oid: 150.0,  # over cap
    )
    with pytest.raises(SpendCapExceeded):
        auth.validate("ar_live_testkey123")


def test_spend_cap_under_limit():
    org = _make_org(monthly_spend_cap=100.0)
    key = _make_key()
    auth = Authenticator(
        store=_store_fn(org, key),
        get_monthly_spend=lambda oid: 50.0,
    )
    # Should not raise
    auth.validate("ar_live_testkey123")


# ── Rate limiting ────────────────────────────────────────────────────

def test_rate_limit_enforced():
    org = _make_org()
    key = _make_key(rate_limit_rpm=5)
    auth = Authenticator(store=_store_fn(org, key))
    # First 5 should pass
    for _ in range(5):
        auth.validate("ar_live_testkey123")
    # 6th should fail
    with pytest.raises(ProviderRateLimitError):
        auth.validate("ar_live_testkey123")


# ── Test key daily limit ─────────────────────────────────────────────

def test_test_key_daily_limit():
    org = _make_org()
    key = ApiKey(
        org_id="org-1",
        key_hash=hash_key("ar_test_testkey123"),
        key_prefix="ar_test_testkey1",
        rate_limit_rpm=10000,  # high RPM so only daily limit triggers
    )
    auth = Authenticator(store=lambda kh: (org, key) if kh == key.key_hash else None)
    # Should work for 100 calls
    for _ in range(100):
        auth.validate("ar_test_testkey123")
    # 101st should fail
    with pytest.raises(ProviderRateLimitError, match="daily"):
        auth.validate("ar_test_testkey123")


# ── Cache ────────────────────────────────────────────────────────────

def test_key_cache_hit():
    cache = _KeyCache(ttl_seconds=60)
    org = _make_org()
    key = _make_key()
    cache.put("hash1", org, key)
    result = cache.get("hash1")
    assert result is not None
    assert result[0].id == "org-1"


def test_key_cache_miss():
    cache = _KeyCache()
    assert cache.get("nonexistent") is None


def test_key_cache_expires():
    cache = _KeyCache(ttl_seconds=0)  # immediate expiry
    org = _make_org()
    key = _make_key()
    cache.put("hash1", org, key)
    import time
    time.sleep(0.01)
    assert cache.get("hash1") is None
