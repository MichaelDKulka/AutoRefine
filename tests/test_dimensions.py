"""Tests for the dimensions module — validation, normalization, aggregation."""

import pytest

from autorefine.dimensions import (
    DimensionAggregator,
    FeedbackDimension,
    FeedbackDimensionSchema,
)
from autorefine.models import FeedbackSignal, FeedbackType


# ── Dimension name validation ─────────────────────────────────────────

def test_valid_dimension_name():
    d = FeedbackDimension(name="accuracy", description="test")
    assert d.name == "accuracy"


def test_dimension_name_with_underscores():
    d = FeedbackDimension(name="response_quality", description="test")
    assert d.name == "response_quality"


def test_invalid_dimension_name_uppercase():
    with pytest.raises(ValueError):
        FeedbackDimension(name="Accuracy", description="test")


def test_invalid_dimension_name_spaces():
    with pytest.raises(ValueError):
        FeedbackDimension(name="my dim", description="test")


def test_reserved_dimension_name():
    with pytest.raises(ValueError):
        FeedbackDimension(name="score", description="test")

    with pytest.raises(ValueError):
        FeedbackDimension(name="overall", description="test")


def test_dimension_name_too_long():
    with pytest.raises(ValueError):
        FeedbackDimension(name="a" * 65, description="test")


# ── Scale validation ──────────────────────────────────────────────────

def test_valid_scale():
    d = FeedbackDimension(name="tone", description="t", scale=(0, 5))
    assert d.scale == (0, 5)


def test_invalid_scale_reversed():
    with pytest.raises(ValueError):
        FeedbackDimension(name="tone", description="t", scale=(5, 0))


# ── Schema normalization ─────────────────────────────────────────────

def _make_schema():
    return FeedbackDimensionSchema.from_dict("test", {
        "accuracy": {
            "description": "Accuracy",
            "scale": (-1.0, 1.0),
            "weight": 2.0,
            "refinement_priority": "high",
        },
        "tone": {
            "description": "Tone",
            "scale": (0.0, 5.0),
            "weight": 1.0,
            "refinement_priority": "low",
        },
    })


def test_normalize_default_scale():
    schema = _make_schema()
    assert schema.normalize_score("accuracy", 0.5) == pytest.approx(0.5)
    assert schema.normalize_score("accuracy", -1.0) == pytest.approx(-1.0)
    assert schema.normalize_score("accuracy", 1.0) == pytest.approx(1.0)


def test_normalize_custom_scale():
    schema = _make_schema()
    # 0 on [0, 5] → -1.0 on [-1, 1]
    assert schema.normalize_score("tone", 0.0) == pytest.approx(-1.0)
    # 5 on [0, 5] → +1.0 on [-1, 1]
    assert schema.normalize_score("tone", 5.0) == pytest.approx(1.0)
    # 2.5 on [0, 5] → 0.0 on [-1, 1]
    assert schema.normalize_score("tone", 2.5) == pytest.approx(0.0)


def test_normalize_clamps():
    schema = _make_schema()
    # Out of range on custom scale → clamped
    assert schema.normalize_score("tone", -1.0) == pytest.approx(-1.0)
    assert schema.normalize_score("tone", 10.0) == pytest.approx(1.0)


# ── Composite score ──────────────────────────────────────────────────

def test_composite_full_dimensions():
    schema = _make_schema()
    # accuracy=1.0 (weight 2), tone=5.0 → normalized 1.0 (weight 1)
    composite = schema.compute_composite({"accuracy": 1.0, "tone": 5.0})
    # weighted avg: (2*1.0 + 1*1.0) / 3 = 1.0
    assert composite == pytest.approx(1.0)


def test_composite_partial_dimensions():
    schema = _make_schema()
    # Only accuracy scored → only accuracy weight matters
    composite = schema.compute_composite({"accuracy": -0.5})
    assert composite == pytest.approx(-0.5)


def test_composite_empty():
    schema = _make_schema()
    assert schema.compute_composite({}) == 0.0


def test_composite_mixed():
    schema = _make_schema()
    # accuracy=-1.0 (weight 2, normalized=-1), tone=2.5 (weight 1, normalized=0)
    composite = schema.compute_composite({"accuracy": -1.0, "tone": 2.5})
    # weighted avg: (2*(-1) + 1*0) / 3 = -0.667
    assert composite == pytest.approx(-2.0 / 3.0, abs=0.01)


# ── Normalized weights ───────────────────────────────────────────────

def test_normalized_weights():
    schema = _make_schema()
    weights = schema.normalized_weights()
    assert weights["accuracy"] == pytest.approx(2.0 / 3.0)
    assert weights["tone"] == pytest.approx(1.0 / 3.0)
    assert sum(weights.values()) == pytest.approx(1.0)


# ── from_dict ────────────────────────────────────────────────────────

def test_from_dict():
    schema = FeedbackDimensionSchema.from_dict("my_key", {
        "accuracy": {"description": "Is it correct?", "weight": 2.0},
        "reasoning": {"description": "Is the reasoning good?"},
    })
    assert schema.prompt_key == "my_key"
    assert len(schema.dimensions) == 2
    assert schema.dimensions["accuracy"].weight == 2.0
    assert schema.dimensions["reasoning"].weight == 1.0
    assert schema.dimensions["reasoning"].refinement_priority == "medium"


# ── Aggregator ───────────────────────────────────────────────────────

def _make_signals(n_positive: int, n_negative: int):
    signals = []
    for _ in range(n_positive):
        signals.append(FeedbackSignal(
            interaction_id="ix1",
            feedback_type=FeedbackType.POSITIVE,
            score=0.8,
            dimensions={"accuracy": 0.9, "tone": 4.0},
        ))
    for _ in range(n_negative):
        signals.append(FeedbackSignal(
            interaction_id="ix2",
            feedback_type=FeedbackType.NEGATIVE,
            score=-0.7,
            dimensions={"accuracy": -0.8, "tone": 1.0},
        ))
    return signals


def test_aggregator_basic():
    schema = _make_schema()
    aggregator = DimensionAggregator(schema)
    signals = _make_signals(7, 5)  # 12 total, >= 5 per dimension
    summaries = aggregator.aggregate(signals)

    assert "accuracy" in summaries
    assert "tone" in summaries
    assert summaries["accuracy"].signal_count == 12
    assert summaries["tone"].signal_count == 12


def test_aggregator_insufficient_data():
    schema = _make_schema()
    aggregator = DimensionAggregator(schema)
    signals = _make_signals(2, 1)  # 3 total, < 5
    summaries = aggregator.aggregate(signals)

    assert summaries["accuracy"].signal_count == 3
    assert summaries["accuracy"].trend == "unknown"


def test_aggregator_format_for_meta_prompt():
    schema = _make_schema()
    aggregator = DimensionAggregator(schema)
    signals = _make_signals(7, 5)
    text = aggregator.format_for_meta_prompt(signals)

    assert "DIMENSION ANALYSIS" in text
    assert "accuracy" in text
    assert "tone" in text
    assert "COMPOSITE" in text
