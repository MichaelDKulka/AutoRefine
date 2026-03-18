"""Usage metering, markup calculation, and Stripe billing sync.

Tracks all LLM calls through the cloud proxy, applies the org's
markup rate, and optionally syncs usage to Stripe for metered billing.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from autorefine.cloud.models import DailyUsageSummary, Organization, UsageRecord
from autorefine.cost_tracker import estimate_cost

logger = logging.getLogger("autorefine.cloud.billing")

# Markup rates by plan
MARKUP_RATES: dict[str, float] = {
    "internal": float(os.environ.get("MARKUP_RATE_INTERNAL", "0.0")),
    "client": float(os.environ.get("MARKUP_RATE_CLIENT", "0.10")),
    "self_serve": float(os.environ.get("MARKUP_RATE_SELF_SERVE", "0.35")),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_markup_rate(org: Organization) -> float:
    """Return the effective markup rate for an organization.

    Uses the org's explicit ``markup_rate`` field, falling back to
    the plan-based default.
    """
    if org.markup_rate is not None:
        return org.markup_rate
    return MARKUP_RATES.get(org.plan, 0.35)


def calculate_cost(
    org: Organization,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float, float]:
    """Calculate upstream cost, markup, and customer cost.

    Returns:
        ``(upstream_cost, markup_amount, customer_cost)``
    """
    upstream = estimate_cost(model, input_tokens, output_tokens)
    rate = get_markup_rate(org)
    markup = upstream * rate
    customer = upstream + markup
    return upstream, markup, customer


class BillingManager:
    """Manages usage metering and billing.

    Args:
        save_record: Callable that persists a UsageRecord.
        get_records: Callable that returns UsageRecords for an org_id
            within a date range.
    """

    def __init__(
        self,
        save_record: Any = None,
        get_records: Any = None,
    ) -> None:
        self._save_record = save_record
        self._get_records = get_records

    def record_usage(
        self,
        org: Organization,
        api_key_id: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        interaction_id: str = "",
        prompt_key: str = "default",
    ) -> UsageRecord:
        """Record a metered usage event."""
        upstream, markup, customer = calculate_cost(
            org, model, input_tokens, output_tokens,
        )

        record = UsageRecord(
            org_id=org.id,
            api_key_id=api_key_id,
            interaction_id=interaction_id,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            upstream_cost=upstream,
            markup_amount=markup,
            customer_cost=customer,
            prompt_key=prompt_key,
            created_at=_utc_now(),
        )

        if self._save_record:
            try:
                self._save_record(record)
            except Exception:
                logger.warning(
                    "Failed to save usage record for org %s", org.id,
                    exc_info=True,
                )

        return record

    def get_monthly_spend(self, org_id: str) -> float:
        """Return the current month's total customer spend for an org."""
        if self._get_records is None:
            return 0.0

        now = _utc_now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        records = self._get_records(org_id, since=month_start)
        return sum(r.customer_cost for r in records)

    def get_daily_breakdown(
        self,
        org_id: str,
        days: int = 30,
    ) -> list[DailyUsageSummary]:
        """Return aggregated daily usage for the last N days."""
        if self._get_records is None:
            return []

        from datetime import timedelta
        since = _utc_now() - timedelta(days=days)
        records = self._get_records(org_id, since=since)

        # Group by date
        daily: dict[str, list[UsageRecord]] = defaultdict(list)
        for r in records:
            day = r.created_at.strftime("%Y-%m-%d")
            daily[day].append(r)

        summaries = []
        for date, recs in sorted(daily.items()):
            models_used: dict[str, int] = defaultdict(int)
            for r in recs:
                models_used[r.model] += 1
            summaries.append(DailyUsageSummary(
                org_id=org_id,
                date=date,
                total_calls=len(recs),
                total_input_tokens=sum(r.input_tokens for r in recs),
                total_output_tokens=sum(r.output_tokens for r in recs),
                total_upstream_cost=sum(r.upstream_cost for r in recs),
                total_markup=sum(r.markup_amount for r in recs),
                total_customer_cost=sum(r.customer_cost for r in recs),
                models_used=dict(models_used),
            ))
        return summaries

    def check_spend_cap(self, org_id: str, cap: float) -> bool:
        """Return False if the org has exceeded their monthly cap."""
        current = self.get_monthly_spend(org_id)
        return current < cap

    def sync_to_stripe(self, org_id: str, stripe_customer_id: str) -> bool:
        """Report usage to Stripe's metered billing API.

        Returns True if sync succeeded, False if Stripe is not configured.
        """
        stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")
        if not stripe_key:
            logger.debug("Stripe not configured — skipping sync for org %s", org_id)
            return False

        try:
            import stripe
            stripe.api_key = stripe_key

            spend = self.get_monthly_spend(org_id)
            # Report as usage-based billing event
            stripe.billing.MeterEvent.create(
                event_name="autorefine_usage",
                payload={
                    "stripe_customer_id": stripe_customer_id,
                    "value": str(int(spend * 100)),  # cents
                },
            )
            logger.info("Synced $%.2f usage to Stripe for org %s", spend, org_id)
            return True
        except ImportError:
            logger.warning("stripe package not installed — skipping sync")
            return False
        except Exception:
            logger.error("Stripe sync failed for org %s", org_id, exc_info=True)
            return False
