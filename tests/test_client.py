"""Tests for the main AutoRefine client and @autorefine decorator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autorefine import AutoRefine, autorefine
from autorefine.models import FeedbackType
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.storage.json_store import JSONStore


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(self, response_text: str = "Mock response") -> None:
        self._response_text = response_text

    def chat(self, system_prompt, messages, **kwargs):
        return ProviderResponse(
            text=self._response_text, input_tokens=10,
            output_tokens=20, model="mock-model",
        )

    def stream(self, system_prompt, messages, **kwargs):
        for word in self._response_text.split():
            yield word + " "

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001


@pytest.fixture
def tmp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield JSONStore(str(Path(tmpdir) / "test_store.json"))


@pytest.fixture
def client(tmp_store):
    with patch("autorefine.client.get_provider", return_value=MockProvider()):
        c = AutoRefine(api_key="test-key", model="mock", store=tmp_store)
        c._provider = MockProvider()
        c._interceptor._provider = MockProvider()
        yield c


# ── Init ─────────────────────────────────────────────────────────────

class TestInit:
    def test_creates_with_defaults(self, tmp_store):
        with patch("autorefine.client.get_provider", return_value=MockProvider()):
            c = AutoRefine(api_key="test", store=tmp_store)
            assert c._cfg.api_key == "test"

    def test_auto_learn_flag(self, tmp_store):
        with patch("autorefine.client.get_provider", return_value=MockProvider()):
            c = AutoRefine(api_key="test", store=tmp_store, auto_learn=True)
            assert c._cfg.auto_learn is True

    def test_config_overrides(self, tmp_store):
        with patch("autorefine.client.get_provider", return_value=MockProvider()):
            c = AutoRefine(api_key="test", store=tmp_store,
                           refine_threshold=5, ab_test_split=0.3)
            assert c._cfg.refine_threshold == 5
            assert c._cfg.ab_test_split == 0.3


# ── Chat & Complete ──────────────────────────────────────────────────

class TestChatAndComplete:
    def test_chat_returns_response(self, client):
        resp = client.chat("Be helpful", [{"role": "user", "content": "Hello"}])
        assert resp.text == "Mock response"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 20
        assert resp.id  # has an interaction ID

    def test_complete_returns_response(self, client):
        resp = client.complete("Be helpful", "Hello")
        assert resp.text == "Mock response"

    def test_chat_logs_interaction(self, client, tmp_store):
        resp = client.chat("", [{"role": "user", "content": "Test"}])
        ix = tmp_store.get_interaction(resp.id)
        assert ix is not None
        assert ix.response_text == "Mock response"

    def test_stream_yields_chunks(self, client):
        chunks = list(client.stream("", [{"role": "user", "content": "Hi"}]))
        assert len(chunks) > 0
        assert "Mock" in "".join(chunks)

    def test_prompt_key_override(self, client, tmp_store):
        resp = client.chat("", [{"role": "user", "content": "Hi"}],
                           prompt_key="billing")
        ix = tmp_store.get_interaction(resp.id)
        assert ix.prompt_key == "billing"

    def test_response_has_cost(self, client):
        resp = client.complete("", "Hello")
        assert resp.cost_usd >= 0

    def test_response_has_model(self, client):
        resp = client.complete("", "Hello")
        assert resp.model == "mock-model"


# ── Feedback ─────────────────────────────────────────────────────────

class TestFeedback:
    def test_submit_positive(self, client):
        resp = client.complete("", "Hello")
        fb = client.feedback(resp.id, "positive")
        assert fb.score == 1.0
        assert fb.feedback_type == FeedbackType.POSITIVE

    def test_submit_negative_with_comment(self, client):
        resp = client.complete("", "Hello")
        fb = client.feedback(resp.id, "negative", comment="Too verbose")
        assert fb.score == -1.0
        assert fb.comment == "Too verbose"

    def test_submit_thumbs_up(self, client):
        resp = client.complete("", "Hello")
        fb = client.feedback(resp.id, "thumbs_up")
        assert fb.score == 1.0

    def test_submit_correction(self, client):
        resp = client.complete("", "Hello")
        fb = client.feedback(resp.id, "correction", comment="Better answer")
        assert fb.feedback_type == FeedbackType.CORRECTION


# ── Prompt management ────────────────────────────────────────────────

class TestPromptManagement:
    def test_set_system_prompt(self, client):
        pv = client.set_system_prompt("You are a chef.")
        assert pv.version == 1
        assert pv.system_prompt == "You are a chef."
        assert pv.is_active is True

    def test_get_active_prompt(self, client):
        client.set_system_prompt("Version 1")
        pv = client.get_active_prompt()
        assert pv is not None
        assert pv.system_prompt == "Version 1"

    def test_get_prompt_history(self, client):
        client.set_system_prompt("V1")
        client.set_system_prompt("V2")
        history = client.get_prompt_history()
        assert len(history) == 2

    def test_rollback(self, client):
        client.set_system_prompt("V1")
        client.set_system_prompt("V2")
        rolled = client.rollback(1)
        assert rolled.version == 1
        active = client.get_active_prompt()
        assert active.version == 1

    def test_rollback_nonexistent_raises(self, client):
        from autorefine.exceptions import RollbackError
        with pytest.raises(RollbackError):
            client.rollback(999)


# ── Analytics & costs ────────────────────────────────────────────────

class TestAnalytics:
    def test_analytics_property(self, client):
        assert client.analytics is not None

    def test_get_analytics(self, client):
        snapshot = client.get_analytics(days=7)
        assert snapshot.total_interactions == 0

    def test_costs_property(self, client):
        costs = client.costs
        assert "monthly_refiner_spend" in costs


# ── Health check ─────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_check_ok(self, client):
        status = client.health_check()
        assert status["ok"] is True
        assert status["store"] == "ok"
        assert status["provider"] == "ok"
        assert status["refiner"] == "not_configured"

    def test_health_check_store_failure(self, tmp_store):
        broken_store = MagicMock()
        broken_store.get_active_prompt.side_effect = RuntimeError("db down")
        with patch("autorefine.client.get_provider", return_value=MockProvider()):
            c = AutoRefine(api_key="test", store=broken_store)
            c._provider = MockProvider()
            c._interceptor._provider = MockProvider()
            status = c.health_check()
            assert status["ok"] is False
            assert "error" in status["store"]


# ── End-to-end ───────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_flow(self, tmp_store):
        with patch("autorefine.client.get_provider", return_value=MockProvider()):
            c = AutoRefine(api_key="test", store=tmp_store)
            c._provider = MockProvider()
            c._interceptor._provider = MockProvider()

            c.set_system_prompt("You are a cooking assistant.")
            resp = c.chat(
                "You are a cooking assistant.",
                [{"role": "user", "content": "How to boil water?"}],
            )
            assert resp.text == "Mock response"

            c.feedback(resp.id, "negative", comment="Too basic")

            ix = tmp_store.get_interaction(resp.id)
            assert ix is not None
            fb_list = tmp_store.get_feedback(unprocessed_only=True)
            assert len(fb_list) == 1
            assert fb_list[0].comment == "Too basic"

    def test_chat_feedback_refine_improved_prompt(self, tmp_store):
        """End-to-end: chat → accumulate feedback → refine → prompt improves."""
        import json


        refiner_response = json.dumps({
            "new_prompt": "You are an expert cooking assistant. Provide detailed steps.",
            "changelog": "Added detail requirement based on user feedback",
            "reasoning": "Users complained responses were too basic",
            "gaps_identified": ["Missing instruction for detail level"],
            "expected_improvements": ["More detailed answers"],
        })

        class RefinerMockProvider(MockProvider):
            """Returns the refiner JSON when called by the refiner."""
            def chat(self, system_prompt, messages, **kwargs):
                if "prompt engineer" in system_prompt.lower():
                    return ProviderResponse(text=refiner_response, input_tokens=500, output_tokens=300, model="mock-refiner")
                return super().chat(system_prompt, messages, **kwargs)

        with patch("autorefine.client.get_provider", return_value=RefinerMockProvider()):
            c = AutoRefine(api_key="test", store=tmp_store, refiner_key="refiner-key",
                           refine_threshold=3, ab_test_split=0.0)
            c._provider = RefinerMockProvider()
            c._interceptor._provider = RefinerMockProvider()
            # Patch the refiner provider too
            c._refiner._provider = RefinerMockProvider()

            c.set_system_prompt("You are a cooking assistant.")

            # Step 1: Chat and collect negative feedback
            for i in range(3):
                resp = c.chat("", [{"role": "user", "content": f"Question {i}"}])
                c.feedback(resp.id, "negative", comment=f"Too basic {i}")

            # Step 2: Trigger refinement manually
            version = c.refine_now()
            assert version is not None
            assert version.version == 2

            # Step 3: Verify the prompt was improved
            active = c.get_active_prompt()
            assert active is not None
            assert "expert cooking assistant" in active.system_prompt
            assert "detailed steps" in active.system_prompt

            # Step 4: Verify feedback was marked processed
            unprocessed = tmp_store.get_feedback(unprocessed_only=True)
            assert len(unprocessed) == 0

            # Step 5: Verify history
            history = c.get_prompt_history()
            assert len(history) == 2
            assert history[0].system_prompt == "You are a cooking assistant."
            assert "expert" in history[1].system_prompt


# ── @autorefine decorator ───────────────────────────────────────────

class TestDecorator:
    def test_decorator_wraps_function(self, tmp_store):
        with patch("autorefine.client.get_provider", return_value=MockProvider()):
            @autorefine(api_key="test", model="mock",
                        store_path=str(tmp_store._path))
            def ask(system: str, prompt: str):
                return {"system": system, "prompt": prompt}

            # Patch the provider on the lazily-created client
            with patch("autorefine.client.get_provider", return_value=MockProvider()):
                resp = ask("Be helpful", "Hi")
                assert resp.text == "Mock response"
                assert resp.id

    def test_decorator_has_feedback_method(self, tmp_store):
        with patch("autorefine.client.get_provider", return_value=MockProvider()):
            @autorefine(api_key="test", model="mock",
                        store_path=str(tmp_store._path))
            def ask(system: str, prompt: str):
                pass

            with patch("autorefine.client.get_provider", return_value=MockProvider()):
                resp = ask("Be helpful", "Hi")
                fb = ask.feedback(resp.id, "thumbs_up")
                assert fb.score == 1.0

    def test_decorator_preserves_name(self):
        @autorefine(api_key="test")
        def my_function(system: str, prompt: str):
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."
