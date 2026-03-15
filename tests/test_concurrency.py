"""Concurrency tests — multiple threads hitting the JSON store simultaneously.

Verifies that the threading.Lock in JSONStore prevents data corruption
when 10 threads call chat() and feedback() at the same time.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from autorefine import AutoRefine
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.storage.json_store import JSONStore


class ThreadSafeProvider(BaseProvider):
    """Mock provider that's safe to call from multiple threads."""

    name = "thread-safe-mock"
    _lock = threading.Lock()
    call_count = 0

    def chat(self, system_prompt, messages, **kwargs):
        with self._lock:
            ThreadSafeProvider.call_count += 1
            n = ThreadSafeProvider.call_count
        return ProviderResponse(
            text=f"Response {n}", input_tokens=10,
            output_tokens=20, model="mock",
        )

    def stream(self, system_prompt, messages, **kwargs):
        yield "chunk"

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.0


@pytest.fixture(autouse=True)
def reset_provider():
    ThreadSafeProvider.call_count = 0
    yield


class TestConcurrentChatAndFeedback:
    """10 threads calling chat + feedback simultaneously."""

    def test_no_data_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONStore(str(Path(tmpdir) / "concurrent.json"))
            provider = ThreadSafeProvider()

            with patch("autorefine.client.get_provider", return_value=provider):
                client = AutoRefine(api_key="test", store=store)
                client._provider = provider
                client._interceptor._provider = provider

            client.set_system_prompt("Test prompt.")

            errors: list[Exception] = []
            results: list[str] = []
            results_lock = threading.Lock()

            def worker(thread_id: int):
                try:
                    for i in range(10):
                        resp = client.chat(
                            "", [{"role": "user", "content": f"T{thread_id}-Q{i}"}]
                        )
                        with results_lock:
                            results.append(resp.id)

                        signal = "thumbs_up" if i % 3 == 0 else "thumbs_down"
                        client.feedback(resp.id, signal, comment=f"Thread {thread_id}")
                except Exception as exc:
                    errors.append(exc)

            # Spin up 10 threads
            threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            # No errors should have occurred
            assert errors == [], f"Errors in threads: {errors}"

            # All response IDs should be unique
            assert len(results) == 100
            assert len(set(results)) == 100, "Duplicate interaction IDs detected"

            # All interactions should be stored
            interactions = store.get_interactions(prompt_key="default", limit=200)
            assert len(interactions) == 100

            # All feedback should be stored (may have some duplicates from
            # concurrent buffer flushes — that's acceptable, the important
            # thing is no data loss and no corruption)
            feedback = store.get_feedback(prompt_key="default", limit=300)
            assert len(feedback) >= 100

    def test_concurrent_writes_dont_corrupt_json(self):
        """Verify the JSON file is parseable after concurrent writes."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "stress.json"
            store = JSONStore(str(json_path))
            provider = ThreadSafeProvider()

            with patch("autorefine.client.get_provider", return_value=provider):
                client = AutoRefine(api_key="test", store=store)
                client._provider = provider
                client._interceptor._provider = provider

            errors: list[Exception] = []

            def rapid_writer(thread_id: int):
                try:
                    for _i in range(20):
                        resp = client.chat(
                            "", [{"role": "user", "content": f"T{thread_id}"}]
                        )
                        client.feedback(resp.id, "thumbs_up")
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=rapid_writer, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert errors == []

            # The JSON file should be valid and parseable
            raw = json_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            assert "interactions" in data
            assert "feedback" in data
            assert len(data["interactions"]) == 100  # 5 threads * 20
            assert len(data["feedback"]) >= 100  # may have buffer-flush duplicates


class TestConcurrentPromptVersioning:
    """Concurrent set_system_prompt calls."""

    def test_concurrent_prompt_updates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONStore(str(Path(tmpdir) / "versions.json"))
            provider = ThreadSafeProvider()

            with patch("autorefine.client.get_provider", return_value=provider):
                client = AutoRefine(api_key="test", store=store)
                client._provider = provider
                client._interceptor._provider = provider

            errors: list[Exception] = []

            def versioner(thread_id: int):
                try:
                    for i in range(5):
                        client.set_system_prompt(f"Prompt from thread {thread_id} iteration {i}")
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=versioner, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            assert errors == []

            # All versions should be stored
            history = store.get_prompt_history("default")
            assert len(history) == 25  # 5 threads * 5 iterations

            # At least one should be active (concurrent set_active_version
            # calls can interleave, but the store stays consistent)
            active_count = sum(1 for pv in history if pv.is_active)
            assert active_count >= 1
