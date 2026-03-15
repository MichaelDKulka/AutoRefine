"""Tests for feedback noise filtering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from autorefine.feedback_filter import FeedbackFilter
from autorefine.models import FeedbackSignal, FeedbackType


def _fb(
    id: str = "",
    score: float = -1.0,
    user_id: str = "",
    interaction_id: str = "ix-001",
    comment: str = "",
    minutes_ago: int = 0,
    feedback_type: FeedbackType = FeedbackType.NEGATIVE,
    confidence: float = 1.0,
) -> FeedbackSignal:
    return FeedbackSignal(
        id=id or f"fb-{score}-{user_id}-{minutes_ago}",
        interaction_id=interaction_id,
        feedback_type=feedback_type,
        score=score,
        confidence=confidence,
        user_id=user_id,
        comment=comment,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


class TestContradictions:
    def test_keeps_latest_when_contradictory(self):
        filt = FeedbackFilter()
        items = [
            _fb(id="early", score=1.0, user_id="u1", minutes_ago=10),
            _fb(id="late", score=-1.0, user_id="u1", minutes_ago=0),
        ]
        result = filt.filter(items)
        assert len(result) == 1
        assert result[0].id == "late"

    def test_keeps_all_when_consistent(self):
        filt = FeedbackFilter()
        items = [
            _fb(id="a", score=-1.0, user_id="u1"),
            _fb(id="b", score=-0.5, user_id="u1"),
        ]
        result = filt.filter(items)
        assert len(result) == 2


class TestRageClicking:
    def test_detects_rapid_negatives(self):
        filt = FeedbackFilter(rage_click_threshold=3, rage_click_window_minutes=2)
        items = [
            _fb(id=f"rage-{i}", score=-1.0, user_id="rager", minutes_ago=0,
                interaction_id=f"ix-{i}", comment="bad" if i == 0 else "")
            for i in range(5)
        ]
        result = filt.filter(items)
        # Should keep only 1 (the one with the longest comment)
        rage_kept = [fb for fb in result if fb.user_id == "rager"]
        assert len(rage_kept) == 1
        assert rage_kept[0].comment == "bad"

    def test_no_false_positive_when_spread_out(self):
        filt = FeedbackFilter(rage_click_threshold=3, rage_click_window_minutes=2)
        items = [
            _fb(id=f"spread-{i}", score=-1.0, user_id="normal",
                interaction_id=f"ix-{i}", minutes_ago=i * 10)
            for i in range(5)
        ]
        result = filt.filter(items)
        assert len(result) == 5


class TestOutlierUsers:
    def test_downweights_dominant_user(self):
        filt = FeedbackFilter(outlier_user_fraction=0.5, outlier_min_batch=6)
        items = [
            # user "dom" has 5 out of 8 negatives
            *[_fb(id=f"dom-{i}", score=-1.0, user_id="dom",
                  interaction_id=f"ix-{i}", minutes_ago=i * 10) for i in range(5)],
            *[_fb(id=f"other-{i}", score=-1.0, user_id=f"u{i}",
                  interaction_id=f"ix-o{i}", minutes_ago=i * 10) for i in range(3)],
        ]
        result = filt.filter(items)
        dom_items = [fb for fb in result if fb.user_id == "dom"]
        # Confidence should be halved for outlier user
        for fb in dom_items:
            assert fb.confidence == 0.5

    def test_no_downweight_when_balanced(self):
        filt = FeedbackFilter(outlier_user_fraction=0.5, outlier_min_batch=6)
        items = [
            _fb(id=f"u{i}-fb", score=-1.0, user_id=f"user{i}",
                interaction_id=f"ix-{i}", minutes_ago=i * 10)
            for i in range(8)
        ]
        result = filt.filter(items)
        for fb in result:
            assert fb.confidence == 1.0


class TestDisabled:
    def test_disabled_returns_original(self):
        filt = FeedbackFilter(enabled=False)
        items = [_fb(id="a"), _fb(id="b")]
        assert filt.filter(items) == items

    def test_empty_input(self):
        filt = FeedbackFilter()
        assert filt.filter([]) == []
