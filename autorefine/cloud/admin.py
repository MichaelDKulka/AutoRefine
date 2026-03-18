"""Admin API for organization and key management.

Provides CRUD operations for organizations and API keys. Used by
the cloud management dashboard and CLI tools.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from autorefine.cloud.keys import generate_key, revoke_key, rotate_key
from autorefine.cloud.models import ApiKey, Organization

logger = logging.getLogger("autorefine.cloud.admin")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AdminManager:
    """Manages organizations and API keys.

    Args:
        save_org: Callable to persist an Organization.
        get_org: Callable to retrieve an Organization by id.
        save_key: Callable to persist an ApiKey.
        get_keys_for_org: Callable returning all ApiKeys for an org.
        get_key_by_id: Callable returning an ApiKey by id.
    """

    def __init__(
        self,
        save_org: Any = None,
        get_org: Any = None,
        save_key: Any = None,
        get_keys_for_org: Any = None,
        get_key_by_id: Any = None,
    ) -> None:
        self._save_org = save_org
        self._get_org = get_org
        self._save_key = save_key
        self._get_keys_for_org = get_keys_for_org
        self._get_key_by_id = get_key_by_id

    # ── Organizations ─────────────────────────────────────────────────

    def create_org(
        self,
        name: str,
        slug: str,
        plan: str = "self_serve",
        monthly_spend_cap: float = 500.0,
    ) -> Organization:
        """Create a new organization."""
        markup_rates = {"internal": 0.0, "client": 0.10, "self_serve": 0.35}
        org = Organization(
            name=name,
            slug=slug,
            plan=plan,
            markup_rate=markup_rates.get(plan, 0.35),
            monthly_spend_cap=monthly_spend_cap,
        )
        if self._save_org:
            self._save_org(org)
        logger.info("Created org %s (%s, plan=%s)", org.id[:8], slug, plan)
        return org

    def get_org(self, org_id: str) -> Organization | None:
        if self._get_org:
            return self._get_org(org_id)
        return None

    def deactivate_org(self, org_id: str) -> bool:
        org = self.get_org(org_id)
        if org is None:
            return False
        org.is_active = False
        if self._save_org:
            self._save_org(org)
        return True

    # ── API Keys ──────────────────────────────────────────────────────

    def create_key(
        self,
        org_id: str,
        key_type: str = "live",
        name: str = "",
    ) -> tuple[str, ApiKey]:
        """Generate a new API key for an organization.

        Returns (plaintext_key, api_key_record). The plaintext is shown
        once and never stored.
        """
        plaintext, api_key = generate_key(org_id, key_type=key_type, name=name)
        if self._save_key:
            self._save_key(api_key)
        return plaintext, api_key

    def list_keys(self, org_id: str) -> list[ApiKey]:
        if self._get_keys_for_org:
            return self._get_keys_for_org(org_id)
        return []

    def revoke_key_by_id(self, key_id: str) -> bool:
        if not self._get_key_by_id:
            return False
        api_key = self._get_key_by_id(key_id)
        if api_key is None:
            return False
        revoked = revoke_key(api_key)
        if self._save_key:
            self._save_key(revoked)
        return True

    def rotate_key_by_id(
        self,
        key_id: str,
        key_type: str = "live",
    ) -> tuple[str, ApiKey] | None:
        if not self._get_key_by_id:
            return None
        old_key = self._get_key_by_id(key_id)
        if old_key is None:
            return None
        plaintext, new_key, revoked = rotate_key(old_key, key_type=key_type)
        if self._save_key:
            self._save_key(revoked)
            self._save_key(new_key)
        return plaintext, new_key
