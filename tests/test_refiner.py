"""Tests for the refiner engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from autorefine.exceptions import CostLimitExceeded, NoFeedbackError, RefinementError
from autorefine.feedback import FeedbackBundle
from autorefine.models import (
    FeedbackSignal,
    FeedbackType,
    Interaction,
    Message,
    MessageRole,
    PromptVersion,
)
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.refiner import Refiner
from autorefine.storage.json_store import JSONStore


def _make_refiner_response(
    new_prompt: str = "Improved prompt",
    changelog: str | list = "Fixed issues",
    gaps: list | None = None,
):
    return json.dumps({
        "new_prompt": new_prompt,
        "changelog": changelog,
        "reasoning": "Analysis of feedback patterns: users complained about X, so I added Y.",
        "gaps_identified": gaps or ["Missing instruction for edge case A"],
        "expected_improvements": ["Better accuracy", "Clearer responses"],
    })


class MockRefinerProvider(BaseProvider):
    """Mock provider that records calls for inspection."""

    name = "mock-refiner"

    def __init__(self, response_text: str = "", responses: list[str] | None = None) -> None:
        self._response_text = response_text or _make_refiner_response()
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def chat(self, system_prompt, messages, **kwargs):
        self.last_system_prompt = system_prompt
        if messages:
            self.last_user_prompt = messages[-1].content if hasattr(messages[-1], 'content') else ""
        self._call_count += 1

        if self._responses:
            text = self._responses.pop(0) if self._responses else self._response_text
        else:
            text = self._response_text

        return ProviderResponse(
            text=text,
            input_tokens=100,
            output_tokens=200,
            model="mock-refiner-v1",
        )

    def stream(self, system_prompt, messages, **kwargs):
        yield self._response_text

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.01

    @property
    def call_count(self):
        return self._call_count


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = JSONStore(str(Path(tmpdir) / "test.json"))
        ix = Interaction(
            id="ix-001",
            prompt_key="default",
            prompt_version=1,
            system_prompt="Original prompt",
            messages=[Message(role=MessageRole.USER, content="Hello")],
            response_text="Hi there, how can I help you today?",
        )
        s.save_interaction(ix)
        s.save_prompt_version(
            PromptVersion(version=1, prompt_key="default", system_prompt="Original prompt")
        )
        for i in range(5):
            s.save_feedback(
                FeedbackSignal(
                    id=f"fb-{i}",
                    interaction_id="ix-001",
                    feedback_type=FeedbackType.NEGATIVE,
                    score=-1.0,
                    comment=f"Issue {i}",
                )
            )
        yield s



# ── Core refinement ──────────────────────────────────────────────────

class TestRefine:
    def test_refine_produces_candidate(self, store):
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            batch_size=10,
            validation_count=0,
        )
        candidate = refiner.refine()
        assert candidate.system_prompt == "Improved prompt"
        assert candidate.parent_version == 1
        assert candidate.prompt_key == "default"

    def test_refine_with_explicit_bundles(self, store):
        ix = store.get_interaction("ix-001")
        fb = store.get_feedback(prompt_key="default")
        bundles = [FeedbackBundle(interaction=ix, feedback=fb)]

        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            validation_count=0,
        )
        candidate = refiner.refine(
            prompt_key="default",
            current_prompt="My custom prompt",
            feedback_bundles=bundles,
        )
        assert candidate.system_prompt == "Improved prompt"

    def test_refine_with_explicit_prompt_key(self, store):
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            validation_count=0,
        )
        candidate = refiner.refine(prompt_key="default")
        assert candidate.prompt_key == "default"

    def test_refine_marks_feedback_processed(self, store):
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            validation_count=0,
        )
        refiner.refine()
        unprocessed = store.get_feedback(unprocessed_only=True)
        assert len(unprocessed) == 0

    def test_refine_tracks_cost(self, store):
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            validation_count=0,
        )
        refiner.refine()
        cost = store.get_monthly_refiner_cost()
        assert cost > 0

    def test_refine_preserves_reasoning(self, store):
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            validation_count=0,
        )
        candidate = refiner.refine()
        assert candidate.reasoning != ""

    def test_refine_handles_changelog_as_list(self, store):
        resp = _make_refiner_response(changelog=["Change 1", "Change 2"])
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(response_text=resp),
            store=store,
            validation_count=0,
        )
        candidate = refiner.refine()
        assert "Change 1" in candidate.changelog
        assert "Change 2" in candidate.changelog


# ── Promotion ────────────────────────────────────────────────────────

class TestPromoteCandidate:
    def test_promote_candidate(self, store):
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            validation_count=0,
        )
        candidate = refiner.refine()
        version = refiner.promote_candidate(candidate)
        assert version.version == 2
        assert version.is_active is True
        assert version.system_prompt == "Improved prompt"

    def test_promote_deactivates_previous(self, store):
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            validation_count=0,
        )
        candidate = refiner.refine()
        refiner.promote_candidate(candidate)
        active = store.get_active_prompt("default")
        assert active.version == 2


# ── Error handling ───────────────────────────────────────────────────

class TestErrors:
    def test_no_feedback_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = JSONStore(str(Path(tmpdir) / "test.json"))
            refiner = Refiner(refiner_provider=MockRefinerProvider(), store=s)
            with pytest.raises(NoFeedbackError):
                refiner.refine()

    def test_cost_limit_raises(self, store):
        refiner = Refiner(
            refiner_provider=MockRefinerProvider(),
            store=store,
            cost_limit=0.0,
        )
        with pytest.raises(CostLimitExceeded):
            refiner.refine()

    def test_invalid_json_raises(self, store):
        provider = MockRefinerProvider(response_text="not json at all")
        refiner = Refiner(refiner_provider=provider, store=store, validation_count=0)
        with pytest.raises(RefinementError, match="invalid JSON"):
            refiner.refine()

    def test_missing_new_prompt_raises(self, store):
        provider = MockRefinerProvider(
            response_text=json.dumps({"changelog": "oops"})
        )
        refiner = Refiner(refiner_provider=provider, store=store, validation_count=0)
        with pytest.raises(RefinementError, match="missing required.*new_prompt"):
            refiner.refine()

    def test_empty_new_prompt_raises(self, store):
        provider = MockRefinerProvider(
            response_text=json.dumps({"new_prompt": "   "})
        )
        refiner = Refiner(refiner_provider=provider, store=store, validation_count=0)
        with pytest.raises(RefinementError, match="empty"):
            refiner.refine()

    def test_non_dict_json_raises(self, store):
        provider = MockRefinerProvider(
            response_text=json.dumps(["not", "a", "dict"])
        )
        refiner = Refiner(refiner_provider=provider, store=store, validation_count=0)
        with pytest.raises(RefinementError, match="list.*instead of.*JSON object"):
            refiner.refine()


# ── Response parsing ─────────────────────────────────────────────────

class TestParseRefinerResponse:
    def test_strips_markdown_fences(self):
        text = '```json\n{"new_prompt": "test", "changelog": "fix"}\n```'
        result = Refiner._parse_refiner_response(text)
        assert result.new_prompt == "test"

    def test_strips_triple_backtick_without_lang(self):
        text = '```\n{"new_prompt": "test"}\n```'
        result = Refiner._parse_refiner_response(text)
        assert result.new_prompt == "test"

    def test_extracts_json_from_surrounding_text(self):
        text = 'Here is my analysis:\n{"new_prompt": "test", "changelog": "fix"}\nDone!'
        result = Refiner._parse_refiner_response(text)
        assert result.new_prompt == "test"

    def test_parses_valid_json(self):
        text = _make_refiner_response("new prompt", "changed things")
        result = Refiner._parse_refiner_response(text)
        assert result.new_prompt == "new prompt"
        assert "changed things" in result.changelog
        assert len(result.expected_improvements) == 2

    def test_handles_changelog_as_list(self):
        text = json.dumps({
            "new_prompt": "test",
            "changelog": ["Fixed A", "Added B"],
        })
        result = Refiner._parse_refiner_response(text)
        assert "Fixed A" in result.changelog
        assert "Added B" in result.changelog

    def test_handles_gaps_as_string(self):
        text = json.dumps({
            "new_prompt": "test",
            "gaps_identified": "Only one gap",
        })
        result = Refiner._parse_refiner_response(text)
        assert result.gaps_identified == ["Only one gap"]

    def test_handles_missing_optional_fields(self):
        text = json.dumps({"new_prompt": "minimal"})
        result = Refiner._parse_refiner_response(text)
        assert result.new_prompt == "minimal"
        assert result.changelog == ""
        assert result.reasoning == ""
        assert result.gaps_identified == []
        assert result.expected_improvements == []

    def test_preserves_feedback_summary(self):
        text = json.dumps({"new_prompt": "test"})
        result = Refiner._parse_refiner_response(text, feedback_summary="5 signals")
        assert result.feedback_summary == "5 signals"


# ── Meta-prompt quality ──────────────────────────────────────────────

class TestMetaPrompt:
    def test_meta_prompt_contains_current_prompt(self, store):
        provider = MockRefinerProvider()
        refiner = Refiner(
            refiner_provider=provider, store=store, validation_count=0,
        )
        refiner.refine()
        assert "Original prompt" in provider.last_user_prompt

    def test_meta_prompt_contains_user_message(self, store):
        provider = MockRefinerProvider()
        refiner = Refiner(
            refiner_provider=provider, store=store, validation_count=0,
        )
        refiner.refine()
        assert "Hello" in provider.last_user_prompt

    def test_meta_prompt_contains_feedback(self, store):
        provider = MockRefinerProvider()
        refiner = Refiner(
            refiner_provider=provider, store=store, validation_count=0,
        )
        refiner.refine()
        assert "Issue 0" in provider.last_user_prompt
        assert "NEGATIVE" in provider.last_user_prompt

    def test_meta_prompt_contains_feedback_summary(self, store):
        provider = MockRefinerProvider()
        refiner = Refiner(
            refiner_provider=provider, store=store, validation_count=0,
        )
        refiner.refine()
        assert "Negative:" in provider.last_user_prompt
        assert "100%" in provider.last_user_prompt

    def test_meta_prompt_contains_surgical_instructions(self, store):
        provider = MockRefinerProvider()
        refiner = Refiner(
            refiner_provider=provider, store=store, validation_count=0,
        )
        refiner.refine()
        prompt = provider.last_user_prompt
        assert "surgical" in prompt.lower() or "Patch" in prompt
        assert "conditional" in prompt.lower()

    def test_system_prompt_mentions_json(self, store):
        provider = MockRefinerProvider()
        refiner = Refiner(
            refiner_provider=provider, store=store, validation_count=0,
        )
        refiner.refine()
        assert "JSON" in provider.last_system_prompt


# ── Validation ───────────────────────────────────────────────────────

class TestValidation:
    def test_validation_replays_interactions(self, store):
        provider = MockRefinerProvider(
            responses=[
                _make_refiner_response(),       # refinement call
                "Validation response 1",        # validation replay
            ]
        )
        refiner = Refiner(
            refiner_provider=provider,
            store=store,
            validation_count=1,
        )
        refiner.refine()
        # 1 refinement call + 1 validation call
        assert provider.call_count == 2

    def test_validation_disabled_with_zero(self, store):
        provider = MockRefinerProvider()
        refiner = Refiner(
            refiner_provider=provider,
            store=store,
            validation_count=0,
        )
        refiner.refine()
        assert provider.call_count == 1  # only refinement, no validation

    def test_validation_failure_does_not_block(self, store):
        """Validation errors should be logged but not prevent the candidate."""

        class FailingProvider(MockRefinerProvider):
            def chat(self, system_prompt, messages, **kwargs):
                self._call_count += 1
                if self._call_count == 1:
                    return ProviderResponse(
                        text=_make_refiner_response(),
                        input_tokens=100,
                        output_tokens=200,
                    )
                raise RuntimeError("Validation call failed")

        refiner = Refiner(
            refiner_provider=FailingProvider(),
            store=store,
            validation_count=3,
        )
        # Should NOT raise despite validation failure
        candidate = refiner.refine()
        assert candidate.system_prompt == "Improved prompt"

    def test_validation_tracks_cost(self, store):
        provider = MockRefinerProvider(
            responses=[
                _make_refiner_response(),
                "Validation output",
            ]
        )
        refiner = Refiner(
            refiner_provider=provider,
            store=store,
            validation_count=1,
        )
        refiner.refine()
        cost = store.get_monthly_refiner_cost()
        # Should include both refinement and validation costs
        assert cost == pytest.approx(0.02, abs=0.005)
