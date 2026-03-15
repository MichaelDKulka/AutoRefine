"""Tests for async support — AsyncAutoRefine, async interceptor, async providers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from autorefine import AsyncAutoRefine
from autorefine.interceptor import Interceptor
from autorefine.models import Message, MessageRole, PromptVersion
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.storage.json_store import JSONStore


class MockAsyncProvider(BaseProvider):
    """Mock provider with native async methods."""

    name = "mock-async"

    def chat(self, system_prompt, messages, **kwargs):
        return ProviderResponse(text=f"sync:{system_prompt}", input_tokens=5, output_tokens=10, model="mock")

    def stream(self, system_prompt, messages, **kwargs):
        for word in f"stream:{system_prompt}".split():
            yield word + " "

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001

    async def async_chat(self, system_prompt, messages, **kwargs):
        return ProviderResponse(text=f"async:{system_prompt}", input_tokens=5, output_tokens=10, model="mock-async")

    async def async_stream(self, system_prompt, messages, **kwargs):
        for word in f"astream:{system_prompt}".split():
            yield word + " "


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield JSONStore(str(Path(tmpdir) / "async_test.json"))


# ── Async interceptor ────────────────────────────────────────────────

class TestAsyncInterceptor:
    @pytest.mark.asyncio
    async def test_async_chat(self, store):
        ic = Interceptor(provider=MockAsyncProvider(), store=store, prompt_key="test")
        resp = await ic.async_intercept_call(
            system_prompt="Be helpful",
            messages_or_prompt=[{"role": "user", "content": "Hello"}],
            call_type="chat",
        )
        assert "async:Be helpful" in resp.text
        assert resp.id

    @pytest.mark.asyncio
    async def test_async_complete(self, store):
        ic = Interceptor(provider=MockAsyncProvider(), store=store, prompt_key="test")
        resp = await ic.async_intercept_call(
            system_prompt="Be helpful",
            messages_or_prompt="Hi there",
            call_type="complete",
        )
        assert resp.text is not None
        assert resp.id

    @pytest.mark.asyncio
    async def test_async_stream(self, store):
        ic = Interceptor(provider=MockAsyncProvider(), store=store, prompt_key="test")
        result = await ic.async_intercept_call(
            system_prompt="Be helpful",
            messages_or_prompt=[{"role": "user", "content": "Hi"}],
            call_type="stream",
        )
        chunks = [c async for c in result]
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_async_logs_interaction(self, store):
        ic = Interceptor(provider=MockAsyncProvider(), store=store, prompt_key="test")
        resp = await ic.async_intercept_call(
            system_prompt="",
            messages_or_prompt=[{"role": "user", "content": "Test"}],
        )
        ix = store.get_interaction(resp.id)
        assert ix is not None
        assert ix.prompt_key == "test"

    @pytest.mark.asyncio
    async def test_async_uses_refined_prompt(self, store):
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="test", system_prompt="refined", is_active=True)
        )
        ic = Interceptor(provider=MockAsyncProvider(), store=store, prompt_key="test")
        resp = await ic.async_intercept_call(
            system_prompt="developer original",
            messages_or_prompt=[{"role": "user", "content": "Hi"}],
        )
        assert "async:refined" in resp.text


# ── AsyncAutoRefine client ───────────────────────────────────────────

class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_chat(self, store):
        with patch("autorefine.async_client.get_provider", return_value=MockAsyncProvider()):
            c = AsyncAutoRefine(api_key="test", store=store)
            c._provider = MockAsyncProvider()
            c._interceptor._provider = MockAsyncProvider()
            resp = await c.chat("Be helpful", [{"role": "user", "content": "Hi"}])
            assert resp.text is not None
            assert resp.id

    @pytest.mark.asyncio
    async def test_complete(self, store):
        with patch("autorefine.async_client.get_provider", return_value=MockAsyncProvider()):
            c = AsyncAutoRefine(api_key="test", store=store)
            c._provider = MockAsyncProvider()
            c._interceptor._provider = MockAsyncProvider()
            resp = await c.complete("Be helpful", "Hi")
            assert resp.text is not None

    @pytest.mark.asyncio
    async def test_feedback(self, store):
        with patch("autorefine.async_client.get_provider", return_value=MockAsyncProvider()):
            c = AsyncAutoRefine(api_key="test", store=store)
            c._provider = MockAsyncProvider()
            c._interceptor._provider = MockAsyncProvider()
            resp = await c.complete("", "Hello")
            fb = await c.feedback(resp.id, "thumbs_up")
            assert fb.score == 1.0

    @pytest.mark.asyncio
    async def test_set_system_prompt(self, store):
        with patch("autorefine.async_client.get_provider", return_value=MockAsyncProvider()):
            c = AsyncAutoRefine(api_key="test", store=store)
            pv = await c.set_system_prompt("You are a chef.")
            assert pv.version == 1
            assert pv.system_prompt == "You are a chef."

    @pytest.mark.asyncio
    async def test_rollback(self, store):
        with patch("autorefine.async_client.get_provider", return_value=MockAsyncProvider()):
            c = AsyncAutoRefine(api_key="test", store=store)
            await c.set_system_prompt("V1")
            await c.set_system_prompt("V2")
            rolled = await c.rollback(1)
            assert rolled.version == 1

    @pytest.mark.asyncio
    async def test_get_analytics(self, store):
        with patch("autorefine.async_client.get_provider", return_value=MockAsyncProvider()):
            c = AsyncAutoRefine(api_key="test", store=store)
            snap = await c.get_analytics(days=7)
            assert snap.total_interactions == 0


# ── Provider async fallback ──────────────────────────────────────────

class TestProviderAsyncFallback:
    """Providers without native async should fall back to asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_fallback_async_chat(self):
        class SyncOnlyProvider(BaseProvider):
            name = "sync-only"
            def chat(self, s, m, **kw): return ProviderResponse(text="sync-result")
            def stream(self, s, m, **kw): yield "sync-chunk"
            def estimate_cost(self, i, o): return 0.0

        p = SyncOnlyProvider()
        resp = await p.async_chat("sys", [Message(role=MessageRole.USER, content="hi")])
        assert resp.text == "sync-result"

    @pytest.mark.asyncio
    async def test_fallback_async_complete(self):
        class SyncOnlyProvider(BaseProvider):
            name = "sync-only"
            def chat(self, s, m, **kw): return ProviderResponse(text="sync-complete")
            def stream(self, s, m, **kw): yield "chunk"
            def estimate_cost(self, i, o): return 0.0

        p = SyncOnlyProvider()
        resp = await p.async_complete("sys", "hello")
        assert resp.text == "sync-complete"
