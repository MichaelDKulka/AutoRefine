"""Tests for storage backends."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from autorefine.models import (
    ABTest,
    CostEntry,
    FeedbackSignal,
    FeedbackType,
    Interaction,
    Message,
    MessageRole,
    PromptVersion,
)
from autorefine.storage.json_store import JSONStore
from autorefine.storage.sqlite_store import SQLiteStore


def _utc_now():
    return datetime.now(timezone.utc)


@pytest.fixture(params=["json", "sqlite"])
def store(request):
    with tempfile.TemporaryDirectory() as tmpdir:
        if request.param == "json":
            yield JSONStore(str(Path(tmpdir) / "test.json"))
        else:
            yield SQLiteStore(str(Path(tmpdir) / "test.db"))


# ── Interactions ─────────────────────────────────────────────────────

class TestInteractions:
    def test_save_and_retrieve(self, store):
        ix = Interaction(
            id="ix-001",
            prompt_key="default",
            messages=[Message(role=MessageRole.USER, content="Hello")],
            response_text="Hi",
        )
        store.save_interaction(ix)
        retrieved = store.get_interaction("ix-001")
        assert retrieved is not None
        assert retrieved.response_text == "Hi"

    def test_get_interactions_by_key(self, store):
        store.save_interaction(Interaction(id="a", prompt_key="key1", response_text="1"))
        store.save_interaction(Interaction(id="b", prompt_key="key2", response_text="2"))
        results = store.get_interactions(prompt_key="key1")
        assert len(results) == 1
        assert results[0].id == "a"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_interaction("nonexistent") is None

    def test_get_interactions_since(self, store):
        old = _utc_now() - timedelta(days=10)
        store.save_interaction(Interaction(id="old", prompt_key="default", created_at=old))
        store.save_interaction(Interaction(id="new", prompt_key="default"))
        results = store.get_interactions(since=_utc_now() - timedelta(days=1))
        assert len(results) == 1
        assert results[0].id == "new"

    def test_get_interactions_limit(self, store):
        for i in range(5):
            store.save_interaction(Interaction(id=f"ix-{i}", prompt_key="default"))
        results = store.get_interactions(limit=3)
        assert len(results) == 3


# ── Feedback ─────────────────────────────────────────────────────────

class TestFeedback:
    def test_save_and_retrieve(self, store):
        store.save_interaction(Interaction(id="ix-001", prompt_key="default"))
        store.save_feedback(
            FeedbackSignal(
                id="fb-001",
                interaction_id="ix-001",
                feedback_type=FeedbackType.POSITIVE,
                score=1.0,
            )
        )
        items = store.get_feedback(prompt_key="default")
        assert len(items) == 1
        assert items[0].score == 1.0

    def test_mark_processed(self, store):
        store.save_interaction(Interaction(id="ix-001", prompt_key="default"))
        store.save_feedback(
            FeedbackSignal(
                id="fb-001",
                interaction_id="ix-001",
                feedback_type=FeedbackType.NEGATIVE,
                score=-1.0,
            )
        )
        store.mark_feedback_processed(["fb-001"])
        unprocessed = store.get_feedback(prompt_key="default", unprocessed_only=True)
        assert len(unprocessed) == 0

    def test_mark_empty_list_is_noop(self, store):
        store.mark_feedback_processed([])  # should not raise

    def test_get_feedback_since(self, store):
        store.save_interaction(Interaction(id="ix-001", prompt_key="default"))
        old = _utc_now() - timedelta(days=10)
        store.save_feedback(
            FeedbackSignal(
                id="fb-old", interaction_id="ix-001",
                feedback_type=FeedbackType.POSITIVE, created_at=old,
            )
        )
        store.save_feedback(
            FeedbackSignal(
                id="fb-new", interaction_id="ix-001",
                feedback_type=FeedbackType.NEGATIVE,
            )
        )
        results = store.get_feedback(since=_utc_now() - timedelta(days=1))
        assert len(results) == 1
        assert results[0].id == "fb-new"


# ── Prompt versions ──────────────────────────────────────────────────

class TestPromptVersions:
    def test_save_and_get_active(self, store):
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="default", system_prompt="V1", is_active=True)
        )
        active = store.get_active_prompt("default")
        assert active is not None
        assert active.system_prompt == "V1"

    def test_set_active_version(self, store):
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="default", system_prompt="V1", is_active=True)
        )
        store.save_prompt_version(
            PromptVersion(version=2, prompt_key="default", system_prompt="V2", is_active=False)
        )
        store.set_active_version("default", 2)
        active = store.get_active_prompt("default")
        assert active.version == 2

    def test_set_active_minus_one_deactivates_all(self, store):
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="default", system_prompt="V1", is_active=True)
        )
        store.set_active_version("default", -1)
        assert store.get_active_prompt("default") is None

    def test_prompt_history(self, store):
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="default", system_prompt="V1")
        )
        store.save_prompt_version(
            PromptVersion(version=2, prompt_key="default", system_prompt="V2")
        )
        history = store.get_prompt_history("default")
        assert len(history) == 2

    def test_get_specific_version(self, store):
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="default", system_prompt="V1")
        )
        pv = store.get_prompt_version("default", 1)
        assert pv is not None
        assert pv.system_prompt == "V1"

    def test_rollback_to_version(self, store):
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="default", system_prompt="V1", is_active=False)
        )
        store.save_prompt_version(
            PromptVersion(version=2, prompt_key="default", system_prompt="V2", is_active=True)
        )
        pv = store.rollback_to_version("default", 1)
        assert pv is not None
        assert pv.version == 1
        assert store.get_active_prompt("default").version == 1

    def test_rollback_to_nonexistent_returns_none(self, store):
        assert store.rollback_to_version("default", 999) is None


# ── A/B tests ────────────────────────────────────────────────────────

class TestABTests:
    def test_save_and_get_active(self, store):
        ab = ABTest(
            id="ab-001",
            prompt_key="default",
            control_version=1,
            candidate_version=2,
        )
        store.save_ab_test(ab)
        active = store.get_active_ab_test("default")
        assert active is not None
        assert active.control_version == 1

    def test_update(self, store):
        ab = ABTest(id="ab-001", prompt_key="default", control_version=1, candidate_version=2)
        store.save_ab_test(ab)
        ab.control_interactions = 50
        store.update_ab_test(ab)
        updated = store.get_active_ab_test("default")
        assert updated.control_interactions == 50


# ── Cost tracking ────────────────────────────────────────────────────

class TestCostTracking:
    def test_save_and_get_monthly(self, store):
        store.save_cost_entry(
            CostEntry(id="c-001", cost_usd=1.50, call_type="refiner")
        )
        store.save_cost_entry(
            CostEntry(id="c-002", cost_usd=0.50, call_type="primary")
        )
        monthly = store.get_monthly_refiner_cost()
        assert monthly == pytest.approx(1.50, abs=0.01)


# ── Maintenance ──────────────────────────────────────────────────────

class TestPurge:
    def test_purge_old_data(self, store):
        old_time = _utc_now() - timedelta(days=100)
        store.save_interaction(
            Interaction(id="old", prompt_key="default", created_at=old_time)
        )
        store.save_interaction(
            Interaction(id="new", prompt_key="default")
        )
        purged = store.purge_old_data(_utc_now() - timedelta(days=50))
        assert purged >= 1
        assert store.get_interaction("new") is not None
        assert store.get_interaction("old") is None


# ── Analytics (convenience method on BaseStore) ──────────────────────

class TestAnalytics:
    def test_get_analytics(self, store):
        store.save_interaction(Interaction(id="ix-001", prompt_key="default"))
        store.save_feedback(
            FeedbackSignal(
                id="fb-001", interaction_id="ix-001",
                feedback_type=FeedbackType.POSITIVE, score=1.0,
            )
        )
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="default", system_prompt="V1")
        )
        analytics = store.get_analytics("default", days=30)
        assert analytics["total_interactions"] == 1
        assert analytics["total_feedback"] == 1
        assert analytics["positive_rate"] == 1.0
        assert analytics["active_version"] == 1
        assert analytics["total_versions"] == 1


# ── get_store factory ────────────────────────────────────────────────

class TestGetStore:
    def test_json_default(self):
        from autorefine.storage import get_store
        with tempfile.TemporaryDirectory() as tmpdir:
            store = get_store("json", path=str(Path(tmpdir) / "test.json"))
            assert isinstance(store, JSONStore)

    def test_sqlite(self):
        from autorefine.storage import get_store
        with tempfile.TemporaryDirectory() as tmpdir:
            store = get_store("sqlite", path=str(Path(tmpdir) / "test.db"))
            assert isinstance(store, SQLiteStore)

    def test_unknown_raises(self):
        from autorefine.storage import get_store
        with pytest.raises(ValueError, match="Unknown storage backend"):
            get_store("redis")

    def test_postgres_requires_url(self):
        from autorefine.storage import get_store
        with pytest.raises(ValueError, match="database_url is required"):
            get_store("postgres")
