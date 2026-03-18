"""API key generation, hashing, and management.

Keys are stored as SHA-256 hashes. The plaintext is shown once at
generation time and never stored.

Key format: ``ar_live_{32 random hex chars}`` or ``ar_test_{32 hex chars}``
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Any

from autorefine.cloud.models import ApiKey

logger = logging.getLogger("autorefine.cloud.keys")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_key(plaintext: str) -> str:
    """SHA-256 hash a key for storage."""
    return hashlib.sha256(plaintext.encode()).hexdigest()


def generate_key(
    org_id: str,
    key_type: str = "live",
    name: str = "",
) -> tuple[str, ApiKey]:
    """Generate a new API key.

    Args:
        org_id: Organization this key belongs to.
        key_type: ``"live"`` or ``"test"``.
        name: Human label (e.g. ``"production"``, ``"staging"``).

    Returns:
        A tuple of ``(plaintext_key, api_key_record)``. The plaintext
        is shown once to the user and never stored.
    """
    prefix = f"ar_{key_type}_"
    random_part = secrets.token_hex(16)
    plaintext = f"{prefix}{random_part}"

    record = ApiKey(
        org_id=org_id,
        key_hash=hash_key(plaintext),
        key_prefix=f"{prefix}{random_part[:8]}",
        name=name,
        created_at=_utc_now(),
    )

    logger.info("Generated %s key %s for org %s", key_type, record.key_prefix, org_id)
    return plaintext, record


def revoke_key(api_key: ApiKey) -> ApiKey:
    """Revoke an API key by marking it inactive."""
    api_key.is_active = False
    logger.info("Revoked key %s", api_key.key_prefix)
    return api_key


def rotate_key(
    old_key: ApiKey,
    key_type: str = "live",
    name: str = "",
) -> tuple[str, ApiKey, ApiKey]:
    """Revoke the old key and generate a new one for the same org.

    Returns:
        ``(new_plaintext, new_api_key, revoked_old_key)``
    """
    revoked = revoke_key(old_key)
    plaintext, new_key = generate_key(
        org_id=old_key.org_id,
        key_type=key_type,
        name=name or old_key.name,
    )
    return plaintext, new_key, revoked
