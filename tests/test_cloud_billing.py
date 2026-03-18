"""Tests for cloud billing and markup calculation."""

from __future__ import annotations

import pytest

from autorefine.cloud.billing import BillingManager, calculate_cost, get_markup_rate
from autorefine.cloud.models import Organization, UsageRecord


def _make_org(plan="self_serve", markup_rate=None):
    org = Organization(id="org-1", name="Test", slug="test", plan=plan)
    if markup_rate is not None:
        org.markup_rate = markup_rate
    return org


# ── Markup rates ──────────────────────────────────────────────────────

def test_markup_internal():
    org = _make_org(plan="internal", markup_rate=0.0)
    assert get_markup_rate(org) == 0.0


def test_markup_client():
    org = _make_org(plan="client", markup_rate=0.10)
    assert get_markup_rate(org) == 0.10


def test_markup_self_serve():
    org = _make_org(plan="self_serve", markup_rate=0.35)
    assert get_markup_rate(org) == 0.35


# ── Cost calculation ──────────────────────────────────────────────────

def test_cost_calculation_internal():
    org = _make_org(plan="internal", markup_rate=0.0)
    upstream, markup, customer = calculate_cost(org, "gpt-4o", 1000, 500)
    assert upstream > 0
    assert markup == 0.0
    assert customer == upstream


def test_cost_calculation_client():
    org = _make_org(plan="client", markup_rate=0.10)
    upstream, markup, customer = calculate_cost(org, "gpt-4o", 1000, 500)
    assert markup == pytest.approx(upstream * 0.10)
    assert customer == pytest.approx(upstream * 1.10)


def test_cost_calculation_self_serve():
    org = _make_org(plan="self_serve", markup_rate=0.35)
    upstream, markup, customer = calculate_cost(org, "gpt-4o", 1000, 500)
    assert markup == pytest.approx(upstream * 0.35)
    assert customer == pytest.approx(upstream * 1.35)


# ── Usage recording ──────────────────────────────────────────────────

def test_usage_record_created():
    records = []
    billing = BillingManager(save_record=lambda r: records.append(r))
    org = _make_org()
    record = billing.record_usage(
        org=org, api_key_id="key-1", model="gpt-4o", provider="openai",
        input_tokens=1000, output_tokens=500, interaction_id="ix-1",
    )
    assert record.org_id == "org-1"
    assert record.input_tokens == 1000
    assert record.customer_cost > 0
    assert len(records) == 1


def test_monthly_spend():
    from datetime import datetime, timezone
    records = [
        UsageRecord(org_id="org-1", customer_cost=10.0, created_at=datetime.now(timezone.utc)),
        UsageRecord(org_id="org-1", customer_cost=5.0, created_at=datetime.now(timezone.utc)),
    ]
    billing = BillingManager(get_records=lambda oid, since=None: records)
    assert billing.get_monthly_spend("org-1") == 15.0


def test_spend_cap_check():
    records = []
    billing = BillingManager(get_records=lambda oid, since=None: records)
    assert billing.check_spend_cap("org-1", cap=100.0) is True


def test_spend_cap_exceeded():
    from datetime import datetime, timezone
    records = [
        UsageRecord(org_id="org-1", customer_cost=200.0, created_at=datetime.now(timezone.utc)),
    ]
    billing = BillingManager(get_records=lambda oid, since=None: records)
    assert billing.check_spend_cap("org-1", cap=100.0) is False


def test_daily_breakdown():
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    records = [
        UsageRecord(org_id="org-1", model="gpt-4o", customer_cost=5.0,
                     input_tokens=100, output_tokens=50, created_at=now),
        UsageRecord(org_id="org-1", model="gpt-4o", customer_cost=3.0,
                     input_tokens=80, output_tokens=40, created_at=now),
    ]
    billing = BillingManager(get_records=lambda oid, since=None: records)
    breakdown = billing.get_daily_breakdown("org-1", days=7)
    assert len(breakdown) == 1
    assert breakdown[0].total_calls == 2
    assert breakdown[0].total_customer_cost == 8.0
