"""Cloud-specific data models for organizations, API keys, and billing.

These models are used exclusively by the cloud server. The SDK itself
never imports them — it talks to the cloud via HTTP.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class Organization(BaseModel):
    """An AutoRefine Cloud customer organization."""

    id: str = Field(default_factory=_new_id)
    name: str = ""
    slug: str = ""
    plan: str = Field(
        default="self_serve",
        description="'internal', 'client', or 'self_serve'.",
    )
    markup_rate: float = Field(
        default=0.35,
        description="0.0 for internal, 0.10 for client, 0.35 for self-serve.",
    )
    monthly_spend_cap: float = Field(
        default=500.0,
        description="Monthly spend cap in USD.",
    )
    stripe_customer_id: str = ""
    upstream_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Provider name -> encrypted API key (BYOK).",
    )
    created_at: datetime = Field(default_factory=_utc_now)
    is_active: bool = True


class ApiKey(BaseModel):
    """An AutoRefine API key belonging to an organization."""

    id: str = Field(default_factory=_new_id)
    org_id: str = ""
    key_hash: str = Field(
        description="SHA-256 hash of the full key (never store plaintext).",
    )
    key_prefix: str = Field(
        description="'ar_live_' or 'ar_test_' + first 8 chars (for display).",
    )
    name: str = ""
    model_restrictions: list[str] = Field(
        default_factory=list,
        description="Empty = all models allowed.",
    )
    monthly_spend_cap: float | None = Field(
        default=None,
        description="Per-key cap (overrides org cap if set).",
    )
    rate_limit_rpm: int = Field(
        default=600,
        description="Requests per minute.",
    )
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utc_now)
    last_used_at: datetime | None = None


class UsageRecord(BaseModel):
    """A single metered usage event for billing."""

    id: str = Field(default_factory=_new_id)
    org_id: str = ""
    api_key_id: str = ""
    interaction_id: str = ""
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    upstream_cost: float = 0.0
    markup_amount: float = 0.0
    customer_cost: float = 0.0
    prompt_key: str = "default"
    created_at: datetime = Field(default_factory=_utc_now)


class DailyUsageSummary(BaseModel):
    """Aggregated daily usage for Stripe reporting."""

    org_id: str = ""
    date: str = ""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_upstream_cost: float = 0.0
    total_markup: float = 0.0
    total_customer_cost: float = 0.0
    models_used: dict[str, int] = Field(default_factory=dict)
