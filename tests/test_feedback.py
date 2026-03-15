"""Tests for feedback collection, normalization, batching, and refinement prep."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autorefine.feedback import (
    FeedbackBundle,
    FeedbackCollector,
    confidence_for_type,
    normalise_score,
)
from autorefine.models import (
    FeedbackSignal,
    FeedbackType,
    Interaction,
    Message,
    MessageRole,
)
from autorefine.storage.json_store import JSONStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = JSONStore(str(Path(tmpdir) / "test.json"))
        ix = Interaction(
            id="ix-001",
            prompt_key="default",
            messages=[Message(role=MessageRole.USER, content="Hello")],
            response_text="Hi there",
        )
        s.save_interaction(ix)
        yield s


@pytest.fixture
def store_multi(store):
    """Store with multiple interactions for batch testing."""
    for i in range(2, 6):
        store.save_interaction(
            Interaction(
                id=f"ix-00{i}",
                prompt_key="default",
                messages=[Message(role=MessageRole.USER, content=f"Question {i}")],
                response_text=f"Answer {i}",
            )
        )
    return store


# ── Score normalization ──────────────────────────────────────────────

class TestNormalization:
    def test_positive_maps_to_1(self):
        assert normalise_score(FeedbackType.POSITIVE) == 1.0

    def test_negative_maps_to_minus_1(self):
        assert normalise_score(FeedbackType.NEGATIVE) == -1.0

    def test_correction_maps_to_minus_0_5(self):
        assert normalise_score(FeedbackType.CORRECTION) == -0.5

    def test_implicit_reask(self):
        assert normalise_score(FeedbackType.IMPLICIT_REASK) == -0.4

    def test_implicit_abandon(self):
        assert normalise_score(FeedbackType.IMPLICIT_ABANDON) == -0.3

    def test_custom_score_clamped_high(self):
        assert normalise_score(FeedbackType.POSITIVE, raw_score=2.0) == 1.0

    def test_custom_score_clamped_low(self):
        assert normalise_score(FeedbackType.NEGATIVE, raw_score=-5.0) == -1.0

    def test_custom_score_passthrough(self):
        assert normalise_score(FeedbackType.POSITIVE, raw_score=0.5) == 0.5

    def test_string_signal_names(self):
        assert normalise_score("thumbs_up") == 1.0
        assert normalise_score("thumbs_down") == -1.0
        assert normalise_score("correction") == -0.5


class TestConfidence:
    def test_explicit_feedback_high_confidence(self):
        assert confidence_for_type(FeedbackType.POSITIVE) == 1.0
        assert confidence_for_type(FeedbackType.NEGATIVE) == 1.0

    def test_correction_high_confidence(self):
        assert confidence_for_type(FeedbackType.CORRECTION) == 0.9

    def test_implicit_feedback_lower_confidence(self):
        assert confidence_for_type(FeedbackType.IMPLICIT_REASK) == 0.5
        assert confidence_for_type(FeedbackType.IMPLICIT_ABANDON) == 0.3


# ── record() — the primary API ──────────────────────────────────────

class TestRecord:
    def test_thumbs_up(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "thumbs_up")
        assert fb.score == 1.0
        assert fb.feedback_type == FeedbackType.POSITIVE
        assert fb.interaction_id == "ix-001"

    def test_thumbs_down(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "thumbs_down")
        assert fb.score == -1.0
        assert fb.feedback_type == FeedbackType.NEGATIVE

    def test_correction_stores_comment_as_correction(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "correction", comment="Better answer here")
        assert fb.score == -0.5
        assert fb.feedback_type == FeedbackType.CORRECTION
        assert fb.correction == "Better answer here"
        assert fb.comment == "Better answer here"

    def test_custom_score_overrides_default(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "positive", score=0.3)
        assert fb.score == 0.3

    def test_custom_score_clamped(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "positive", score=5.0)
        assert fb.score == 1.0

    def test_comment_preserved(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "thumbs_down", comment="Too verbose")
        assert fb.comment == "Too verbose"

    def test_user_id_preserved(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "thumbs_up", user_id="user-42")
        assert fb.user_id == "user-42"

    def test_metadata_preserved(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "thumbs_up", metadata={"source": "web"})
        assert fb.metadata["source"] == "web"

    def test_accepts_feedback_type_enum(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", FeedbackType.POSITIVE)
        assert fb.score == 1.0


# ── Batching ─────────────────────────────────────────────────────────

class TestBatching:
    def test_batch_size_1_flushes_immediately(self, store):
        c = FeedbackCollector(store=store, prompt_key="default", batch_size=1)
        c.record("ix-001", "thumbs_up")
        assert c.buffer_size == 0
        assert len(store.get_feedback(prompt_key="default")) == 1

    def test_batch_accumulates_until_full(self, store):
        c = FeedbackCollector(store=store, prompt_key="default", batch_size=3)
        c.record("ix-001", "thumbs_up")
        c.record("ix-001", "thumbs_down")
        assert c.buffer_size == 2
        assert len(store.get_feedback(prompt_key="default")) == 0

        # Third record triggers flush
        c.record("ix-001", "thumbs_up")
        assert c.buffer_size == 0
        assert len(store.get_feedback(prompt_key="default")) == 3

    def test_flush_empties_buffer(self, store):
        c = FeedbackCollector(store=store, prompt_key="default", batch_size=10)
        c.record("ix-001", "thumbs_up")
        c.record("ix-001", "thumbs_down")
        assert c.buffer_size == 2

        flushed = c.flush()
        assert flushed == 2
        assert c.buffer_size == 0
        assert len(store.get_feedback(prompt_key="default")) == 2

    def test_flush_empty_buffer_returns_zero(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        assert c.flush() == 0

    def test_buffer_size_property(self, store):
        c = FeedbackCollector(store=store, prompt_key="default", batch_size=10)
        assert c.buffer_size == 0
        c.record("ix-001", "thumbs_up")
        assert c.buffer_size == 1


# ── submit() backward compat ────────────────────────────────────────

class TestSubmit:
    def test_submit_creates_signal(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.submit("ix-001", "positive")
        assert fb.score == 1.0
        assert fb.interaction_id == "ix-001"

    def test_submit_with_comment(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.submit("ix-001", "negative", comment="Bad response")
        assert fb.comment == "Bad response"
        assert fb.score == -1.0

    def test_submit_stores_in_backend(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        c.submit("ix-001", "positive")
        stored = store.get_feedback(prompt_key="default")
        assert len(stored) == 1

    def test_submit_with_correction_field(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.submit("ix-001", "correction", correction="Better answer")
        assert fb.correction == "Better answer"

    def test_threshold_triggers_callback(self, store):
        callback = MagicMock()
        c = FeedbackCollector(
            store=store, prompt_key="default",
            refine_threshold=3, on_ready=callback,
        )
        for _ in range(3):
            c.submit("ix-001", "negative")
        callback.assert_called()

    def test_no_trigger_below_threshold(self, store):
        callback = MagicMock()
        c = FeedbackCollector(
            store=store, prompt_key="default",
            refine_threshold=10, on_ready=callback,
        )
        c.submit("ix-001", "negative")
        callback.assert_not_called()

    def test_callback_failure_does_not_propagate(self, store):
        def bad_callback():
            raise RuntimeError("boom")

        c = FeedbackCollector(
            store=store, prompt_key="default",
            refine_threshold=1, on_ready=bad_callback,
        )
        # Should NOT raise
        c.submit("ix-001", "negative")

    def test_get_unprocessed_count(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        c.submit("ix-001", "positive")
        c.submit("ix-001", "negative")
        assert c.get_unprocessed_count() == 2


# ── Refinement gating ────────────────────────────────────────────────

class TestRefinementGating:
    def test_should_trigger_when_threshold_met(self, store):
        c = FeedbackCollector(store=store, prompt_key="default", refine_threshold=3)
        for _ in range(3):
            c.record("ix-001", "thumbs_down")
        c.flush()
        assert c.should_trigger_refinement() is True

    def test_should_not_trigger_below_threshold(self, store):
        c = FeedbackCollector(store=store, prompt_key="default", refine_threshold=10)
        c.record("ix-001", "thumbs_down")
        c.flush()
        assert c.should_trigger_refinement() is False

    def test_should_trigger_flushes_buffer_first(self, store):
        c = FeedbackCollector(
            store=store, prompt_key="default",
            refine_threshold=2, batch_size=100,
        )
        c.record("ix-001", "thumbs_down")
        c.record("ix-001", "thumbs_down")
        # Buffer hasn't flushed yet, but should_trigger flushes internally
        assert c.should_trigger_refinement() is True

    def test_should_trigger_with_prompt_key_override(self, store):
        store.save_interaction(
            Interaction(id="ix-other", prompt_key="billing")
        )
        c = FeedbackCollector(store=store, prompt_key="default", refine_threshold=1)
        c.submit("ix-other", "negative")
        # Feedback is on "billing" namespace, not "default"
        assert c.should_trigger_refinement("default") is False
        assert c.should_trigger_refinement("billing") is True


# ── Refinement batch preparation ─────────────────────────────────────

class TestRefinementBatch:
    def test_get_refinement_batch_returns_bundles(self, store_multi):
        c = FeedbackCollector(store=store_multi, prompt_key="default")
        c.record("ix-001", "thumbs_down", comment="Too short")
        c.record("ix-002", "thumbs_up")
        c.record("ix-003", "correction", comment="Better answer")
        c.flush()

        bundles = c.get_refinement_batch()
        assert len(bundles) == 3
        assert all(isinstance(b, FeedbackBundle) for b in bundles)

    def test_bundle_contains_interaction_and_feedback(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        c.record("ix-001", "thumbs_down", comment="Bad")
        c.record("ix-001", "thumbs_up", comment="Actually good")
        c.flush()

        bundles = c.get_refinement_batch()
        assert len(bundles) == 1
        bundle = bundles[0]
        assert bundle.interaction.id == "ix-001"
        assert bundle.interaction.response_text == "Hi there"
        assert len(bundle.feedback) == 2

    def test_batch_respects_limit(self, store_multi):
        c = FeedbackCollector(store=store_multi, prompt_key="default")
        for i in range(1, 6):
            c.record(f"ix-00{i}", "thumbs_down")
        c.flush()

        bundles = c.get_refinement_batch(limit=2)
        # Limit applies to feedback count, not bundle count
        total_fb = sum(len(b.feedback) for b in bundles)
        assert total_fb <= 2

    def test_empty_batch_when_no_feedback(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        bundles = c.get_refinement_batch()
        assert bundles == []

    def test_batch_flushes_buffer_first(self, store):
        c = FeedbackCollector(store=store, prompt_key="default", batch_size=100)
        c.record("ix-001", "thumbs_down")
        # Still in buffer
        assert c.buffer_size == 1

        bundles = c.get_refinement_batch()
        assert len(bundles) == 1
        assert c.buffer_size == 0

    def test_missing_interaction_skipped(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        # Record feedback for a non-existent interaction
        fb = FeedbackSignal(
            interaction_id="ix-gone",
            feedback_type=FeedbackType.NEGATIVE,
            score=-1.0,
        )
        store.save_feedback(fb)

        bundles = c.get_refinement_batch()
        # The feedback for ix-gone should be skipped
        ix_ids = {b.interaction.id for b in bundles}
        assert "ix-gone" not in ix_ids


# ── Mark processed ───────────────────────────────────────────────────

class TestMarkProcessed:
    def test_mark_batch_processed(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        c.record("ix-001", "thumbs_down")
        c.record("ix-001", "thumbs_up")
        c.flush()

        bundles = c.get_refinement_batch()
        assert len(bundles) == 1
        c.mark_batch_processed(bundles)

        # Now unprocessed count should be 0
        assert c.get_unprocessed_count() == 0

    def test_processed_feedback_excluded_from_next_batch(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        c.record("ix-001", "thumbs_down")
        c.flush()

        bundles = c.get_refinement_batch()
        c.mark_batch_processed(bundles)

        # New feedback added after processing
        c.record("ix-001", "thumbs_up")
        c.flush()

        bundles2 = c.get_refinement_batch()
        assert len(bundles2) == 1
        assert len(bundles2[0].feedback) == 1
        assert bundles2[0].feedback[0].score == 1.0  # only the new one

    def test_mark_empty_batch_is_noop(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        c.mark_batch_processed([])  # should not raise


# ── Implicit feedback stub ───────────────────────────────────────────

class TestImplicitFeedbackStub:
    def test_detect_implicit_returns_none(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        result = c.detect_implicit_feedback("ix-001", "same question", 5.0)
        assert result is None

    def test_manual_implicit_signals_still_work(self, store):
        c = FeedbackCollector(store=store, prompt_key="default")
        fb = c.record("ix-001", "implicit_reask")
        assert fb.feedback_type == FeedbackType.IMPLICIT_REASK
        assert fb.score == -0.4
        assert fb.confidence == 0.5
