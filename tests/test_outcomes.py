"""Tests for the outcomes module — outcome translation and confidence extraction."""

import pytest

from autorefine.dimensions import FeedbackDimensionSchema
from autorefine.models import Interaction
from autorefine.outcomes import OutcomeTranslator


def _make_schema():
    return FeedbackDimensionSchema.from_dict("test", {
        "accuracy": {"description": "Accuracy", "weight": 2.0, "refinement_priority": "high"},
        "reasoning": {"description": "Reasoning", "weight": 1.5},
        "calibration": {"description": "Calibration", "weight": 1.5, "refinement_priority": "high"},
        "tone": {"description": "Tone", "weight": 0.5, "refinement_priority": "low"},
    })


def _make_interaction(response_text: str) -> Interaction:
    return Interaction(
        id="test-ix",
        prompt_key="test",
        response_text=response_text,
    )


# ── Correct outcome ──────────────────────────────────────────────────

def test_correct_outcome():
    schema = _make_schema()
    translator = OutcomeTranslator(dimension_schema=schema)
    scores = translator.translate(
        {"predicted": "Rain", "actual": "Rain", "correct": True}
    )
    assert scores["accuracy"] == 1.0
    assert scores["reasoning"] == 0.3
    assert scores["tone"] == 0.3


# ── Incorrect outcome ────────────────────────────────────────────────

def test_incorrect_outcome_no_confidence():
    schema = _make_schema()
    translator = OutcomeTranslator(dimension_schema=schema)
    scores = translator.translate(
        {"predicted": "Rain", "actual": "Sun", "correct": False}
    )
    assert scores["accuracy"] == -1.0
    assert scores["calibration"] == 0.0  # no confidence extractable
    assert scores["reasoning"] == 0.0


# ── Incorrect outcome with confidence ────────────────────────────────

def test_incorrect_outcome_high_confidence():
    schema = _make_schema()
    translator = OutcomeTranslator(dimension_schema=schema)
    ix = _make_interaction("I predict with 90% confidence that it will rain.")
    scores = translator.translate(
        {"predicted": "Rain", "actual": "Sun", "correct": False},
        interaction=ix,
    )
    assert scores["accuracy"] == -1.0
    assert scores["calibration"] < -0.5  # high confidence when wrong


def test_incorrect_outcome_low_confidence():
    schema = _make_schema()
    translator = OutcomeTranslator(dimension_schema=schema)
    ix = _make_interaction("I think there's a 55% chance of rain.")
    scores = translator.translate(
        {"predicted": "Rain", "actual": "Sun", "correct": False},
        interaction=ix,
    )
    assert scores["calibration"] > -0.5  # low confidence when wrong → mild penalty


# ── Dimension overrides ──────────────────────────────────────────────

def test_dimension_overrides():
    schema = _make_schema()
    translator = OutcomeTranslator(dimension_schema=schema)
    scores = translator.translate(
        {"predicted": "A", "actual": "B", "correct": False},
        dimension_overrides={"calibration": -0.9, "tone": 1.0},
    )
    assert scores["calibration"] == -0.9
    assert scores["tone"] == 1.0
    assert scores["accuracy"] == -1.0  # auto-translated


# ── No schema ────────────────────────────────────────────────────────

def test_no_schema():
    translator = OutcomeTranslator()
    scores = translator.translate(
        {"predicted": "A", "actual": "A", "correct": True}
    )
    assert scores["_composite"] == 1.0


def test_no_schema_incorrect():
    translator = OutcomeTranslator()
    scores = translator.translate(
        {"predicted": "A", "actual": "B", "correct": False}
    )
    assert scores["_composite"] == -1.0


# ── Confidence extraction ────────────────────────────────────────────

def test_extract_percentage():
    translator = OutcomeTranslator()
    assert translator._extract_confidence_from_response("85% chance") == pytest.approx(0.85)


def test_extract_probability():
    translator = OutcomeTranslator()
    assert translator._extract_confidence_from_response("probability: 0.7") == pytest.approx(0.7)


def test_extract_confidence_word():
    translator = OutcomeTranslator()
    assert translator._extract_confidence_from_response("confidence: high") == pytest.approx(0.8)
    assert translator._extract_confidence_from_response("confidence: very low") == pytest.approx(0.05)


def test_extract_out_of():
    translator = OutcomeTranslator()
    assert translator._extract_confidence_from_response("7 out of 10") == pytest.approx(0.7)


def test_extract_none():
    translator = OutcomeTranslator()
    assert translator._extract_confidence_from_response("I think it might rain.") is None


def test_extract_empty():
    translator = OutcomeTranslator()
    assert translator._extract_confidence_from_response("") is None
