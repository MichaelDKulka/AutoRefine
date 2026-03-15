"""Tests for the interceptor middleware."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autorefine.interceptor import Interceptor
from autorefine.models import Message, MessageRole, PromptVersion
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.storage.json_store import JSONStore

# ── Mock provider ────────────────────────────────────────────────────

class MockProvider(BaseProvider):
    name = "mock"

    def chat(self, system_prompt, messages, **kwargs):
        return ProviderResponse(
            text=f"echo: {system_prompt}",
            input_tokens=5,
            output_tokens=10,
            model="mock-v1",
            finish_reason="stop",
        )

    def stream(self, system_prompt, messages, **kwargs):
        for word in f"stream: {system_prompt}".split():
            yield word + " "

    def estimate_cost(self, input_tokens, output_tokens):
        return (input_tokens + output_tokens) * 0.0001


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield JSONStore(str(Path(tmpdir) / "test.json"))


@pytest.fixture
def interceptor(store):
    return Interceptor(
        provider=MockProvider(),
        store=store,
        prompt_key="test",
        fallback_system="fallback prompt",
    )


# ── Prompt resolution ────────────────────────────────────────────────

class TestPromptResolution:
    def test_uses_fallback_when_no_refined_prompt(self, interceptor):
        resp = interceptor.complete(
            [Message(role=MessageRole.USER, content="Hello")],
        )
        assert "fallback prompt" in resp.text

    def test_uses_refined_prompt_when_available(self, interceptor, store):
        store.save_prompt_version(
            PromptVersion(
                version=1,
                prompt_key="test",
                system_prompt="refined prompt v1",
                is_active=True,
            )
        )
        resp = interceptor.complete(
            [Message(role=MessageRole.USER, content="Hello")],
        )
        assert "refined prompt v1" in resp.text

    def test_developer_system_prompt_used_when_no_version(self, store):
        ic = Interceptor(provider=MockProvider(), store=store, prompt_key="test")
        resp = ic.complete(
            [Message(role=MessageRole.USER, content="Hello")],
            system="user-provided system",
        )
        assert "user-provided system" in resp.text

    def test_refined_prompt_overrides_developer_prompt(self, store):
        store.save_prompt_version(
            PromptVersion(
                version=1,
                prompt_key="test",
                system_prompt="refined override",
                is_active=True,
            )
        )
        ic = Interceptor(provider=MockProvider(), store=store, prompt_key="test")
        resp = ic.complete(
            [Message(role=MessageRole.USER, content="Hello")],
            system="developer original",
        )
        assert "refined override" in resp.text
        # Original prompt preserved in metadata
        ix = store.get_interaction(resp.id)
        assert ix.metadata["original_system_prompt"] == "developer original"


# ── Non-streaming calls ──────────────────────────────────────────────

class TestNonStreaming:
    def test_complete_returns_response(self, interceptor):
        resp = interceptor.complete(
            [Message(role=MessageRole.USER, content="Hello")],
        )
        assert resp.text is not None
        assert resp.id  # has an interaction ID
        assert resp.input_tokens == 5
        assert resp.output_tokens == 10
        assert resp.model == "mock-v1"

    def test_complete_logs_interaction(self, interceptor, store):
        resp = interceptor.complete(
            [Message(role=MessageRole.USER, content="Hello")],
        )
        ix = store.get_interaction(resp.id)
        assert ix is not None
        assert ix.prompt_key == "test"
        assert ix.response_text == resp.text
        assert ix.input_tokens == 5
        assert ix.output_tokens == 10
        assert ix.cost_usd > 0

    def test_accepts_dict_messages(self, interceptor, store):
        resp = interceptor.complete(
            [{"role": "user", "content": "Hello"}],
        )
        assert resp.text is not None
        ix = store.get_interaction(resp.id)
        assert ix is not None

    def test_cost_estimated_and_recorded(self, interceptor, store):
        resp = interceptor.complete(
            [Message(role=MessageRole.USER, content="Hello")],
        )
        ix = store.get_interaction(resp.id)
        # MockProvider: (5 + 10) * 0.0001 = 0.0015
        assert ix.cost_usd == pytest.approx(0.0015, abs=0.001)
        assert resp.cost_usd == ix.cost_usd

    def test_interaction_id_is_unique(self, interceptor):
        resp1 = interceptor.complete([{"role": "user", "content": "A"}])
        resp2 = interceptor.complete([{"role": "user", "content": "B"}])
        assert resp1.id != resp2.id

    def test_messages_include_assistant_reply(self, interceptor):
        resp = interceptor.complete([{"role": "user", "content": "Hello"}])
        assert resp.messages[-1].role == MessageRole.ASSISTANT
        assert resp.messages[-1].content == resp.text


# ── Streaming calls ──────────────────────────────────────────────────

class TestStreaming:
    def test_stream_yields_chunks(self, interceptor):
        chunks = list(
            interceptor.stream([Message(role=MessageRole.USER, content="Hi")])
        )
        assert len(chunks) > 0
        assert any("stream:" in c or "fallback" in c for c in chunks)

    def test_stream_logs_after_completion(self, interceptor, store):
        chunks = list(
            interceptor.stream([Message(role=MessageRole.USER, content="Hi")])
        )
        interactions = store.get_interactions(prompt_key="test")
        assert len(interactions) == 1
        ix = interactions[0]
        assert ix.response_text == "".join(chunks)
        assert ix.metadata.get("streamed") is True

    def test_stream_logs_on_partial_consumption(self, interceptor, store):
        """If the caller only reads part of the stream, the partial
        interaction is still logged (via the finally block)."""
        gen = interceptor.stream(
            [Message(role=MessageRole.USER, content="Hi")]
        )
        first = next(gen)
        # Explicitly close the generator to trigger finally
        gen.close()
        interactions = store.get_interactions(prompt_key="test")
        assert len(interactions) == 1
        assert first in interactions[0].response_text


# ── intercept_call unified API ───────────────────────────────────────

class TestInterceptCall:
    def test_chat_call_type(self, interceptor):
        resp = interceptor.intercept_call(
            system_prompt="Be helpful",
            messages_or_prompt=[{"role": "user", "content": "Hi"}],
            call_type="chat",
        )
        assert resp.text is not None
        assert resp.id

    def test_complete_call_type_with_string(self, interceptor):
        resp = interceptor.intercept_call(
            system_prompt="Be helpful",
            messages_or_prompt="What is 2+2?",
            call_type="complete",
        )
        assert resp.text is not None

    def test_stream_call_type(self, interceptor):
        chunks = list(
            interceptor.intercept_call(
                system_prompt="Be helpful",
                messages_or_prompt=[{"role": "user", "content": "Hi"}],
                call_type="stream",
            )
        )
        assert len(chunks) > 0

    def test_prompt_key_override(self, interceptor, store):
        store.save_prompt_version(
            PromptVersion(
                version=1,
                prompt_key="billing",
                system_prompt="billing bot",
                is_active=True,
            )
        )
        resp = interceptor.intercept_call(
            prompt_key="billing",
            system_prompt="ignored",
            messages_or_prompt=[{"role": "user", "content": "Hi"}],
        )
        assert "billing bot" in resp.text

    def test_kwargs_forwarded(self, store):
        """Provider kwargs like temperature should pass through."""

        class TrackingProvider(MockProvider):
            last_kwargs = {}

            def chat(self, system_prompt, messages, **kwargs):
                TrackingProvider.last_kwargs = kwargs
                return super().chat(system_prompt, messages, **kwargs)

        ic = Interceptor(provider=TrackingProvider(), store=store, prompt_key="t")
        ic.intercept_call(
            messages_or_prompt="Hi",
            call_type="complete",
            temperature=0.5,
            max_tokens=100,
        )
        assert TrackingProvider.last_kwargs["temperature"] == 0.5
        assert TrackingProvider.last_kwargs["max_tokens"] == 100


# ── Safety — store failures must not break the LLM call ──────────────

class TestSafety:
    def test_store_save_failure_does_not_raise(self, store):
        """If the store fails to save, the response is still returned."""
        broken_store = MagicMock(spec=store)
        broken_store.get_active_prompt.return_value = None
        broken_store.save_interaction.side_effect = RuntimeError("disk full")

        ic = Interceptor(
            provider=MockProvider(), store=broken_store, prompt_key="test"
        )
        # Should NOT raise
        resp = ic.complete([{"role": "user", "content": "Hello"}])
        assert resp.text is not None
        assert resp.id  # still has an interaction ID

    def test_store_lookup_failure_uses_developer_prompt(self, store):
        """If the store fails on prompt lookup, fall back to developer prompt."""
        broken_store = MagicMock(spec=store)
        broken_store.get_active_prompt.side_effect = RuntimeError("db down")
        broken_store.save_interaction.return_value = None

        ic = Interceptor(
            provider=MockProvider(), store=broken_store, prompt_key="test",
            fallback_system="safe fallback",
        )
        resp = ic.complete([{"role": "user", "content": "Hello"}])
        assert "safe fallback" in resp.text

    def test_cost_estimation_failure_returns_zero(self, store):
        """If cost estimation fails, record $0.00 rather than crashing."""

        class BrokenCostProvider(MockProvider):
            def estimate_cost(self, input_tokens, output_tokens):
                raise ValueError("pricing not available")

        ic = Interceptor(
            provider=BrokenCostProvider(), store=store, prompt_key="test"
        )
        resp = ic.complete([{"role": "user", "content": "Hello"}])
        assert resp.cost_usd == 0.0

    def test_stream_store_failure_still_yields(self, store):
        """Streaming should still yield all chunks even if store fails."""
        broken_store = MagicMock(spec=store)
        broken_store.get_active_prompt.return_value = None
        broken_store.save_interaction.side_effect = RuntimeError("disk full")

        ic = Interceptor(
            provider=MockProvider(), store=broken_store, prompt_key="test"
        )
        chunks = list(ic.stream([{"role": "user", "content": "Hi"}]))
        assert len(chunks) > 0

    def test_internal_error_falls_through_to_raw_provider(self, store):
        """If AutoRefine internals break, the call still goes through."""
        broken_store = MagicMock(spec=store)
        # Prompt lookup works, but save_interaction raises a weird error
        broken_store.get_active_prompt.return_value = None
        # Simulate an internal AutoRefine error (not a ProviderError)
        broken_store.save_interaction.side_effect = RuntimeError("internal boom")

        ic = Interceptor(
            provider=MockProvider(), store=broken_store, prompt_key="test"
        )
        # This should NOT raise — it should fall through
        resp = ic.complete([{"role": "user", "content": "Hello"}])
        assert resp.text is not None

    def test_provider_error_propagates(self, store):
        """Provider errors (auth, rate limit) should propagate to the developer."""
        from autorefine.exceptions import ProviderAuthError

        class FailingProvider(MockProvider):
            def chat(self, system_prompt, messages, **kwargs):
                raise ProviderAuthError("Invalid API key", provider="mock")

        ic = Interceptor(
            provider=FailingProvider(), store=store, prompt_key="test"
        )
        with pytest.raises(ProviderAuthError):
            ic.complete([{"role": "user", "content": "Hello"}])
