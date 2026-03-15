"""Tests for prompt-change notifications."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from autorefine.notifications import PromptChangeEvent, PromptChangeNotifier


class TestPromptChangeEvent:
    def test_to_dict(self):
        event = PromptChangeEvent(
            prompt_key="default",
            old_version=1,
            new_version=2,
            changelog="Fixed verbosity",
            trigger="auto_refine",
            old_prompt="Be verbose",
            new_prompt="Be concise by default",
        )
        d = event.to_dict()
        assert d["event"] == "prompt_change"
        assert d["old_version"] == 1
        assert d["new_version"] == 2
        assert d["trigger"] == "auto_refine"
        assert "timestamp" in d

    def test_prompt_preview_truncated(self):
        event = PromptChangeEvent(
            prompt_key="default",
            old_version=1,
            new_version=2,
            changelog="test",
            trigger="manual",
            old_prompt="x" * 500,
            new_prompt="y" * 500,
        )
        d = event.to_dict()
        assert len(d["old_prompt_preview"]) == 200
        assert len(d["new_prompt_preview"]) == 200


class TestPromptChangeNotifier:
    def test_callback_fires(self):
        callback = MagicMock()
        notifier = PromptChangeNotifier(on_prompt_change=callback)
        event = PromptChangeEvent(
            prompt_key="default",
            old_version=1,
            new_version=2,
            changelog="test",
            trigger="auto_refine",
        )
        notifier.notify(event)
        callback.assert_called_once_with(event)

    def test_callback_exception_does_not_propagate(self):
        def bad_callback(event):
            raise RuntimeError("boom")

        notifier = PromptChangeNotifier(on_prompt_change=bad_callback)
        event = PromptChangeEvent(
            prompt_key="default",
            old_version=1,
            new_version=2,
            changelog="test",
            trigger="manual",
        )
        # Should not raise
        notifier.notify(event)

    @patch("autorefine.notifications.httpx.post")
    def test_webhook_fires(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        notifier = PromptChangeNotifier(webhook_url="https://hooks.example.com/alert")
        event = PromptChangeEvent(
            prompt_key="default",
            old_version=1,
            new_version=2,
            changelog="test",
            trigger="auto_refine",
        )
        notifier.notify(event)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"]["event"] == "prompt_change"

    @patch("autorefine.notifications.httpx.post")
    def test_webhook_failure_logged_not_raised(self, mock_post):
        mock_post.side_effect = Exception("network error")
        notifier = PromptChangeNotifier(webhook_url="https://hooks.example.com/alert")
        event = PromptChangeEvent(
            prompt_key="default",
            old_version=1,
            new_version=2,
            changelog="test",
            trigger="auto_refine",
        )
        # Should not raise
        notifier.notify(event)

    def test_no_webhook_no_callback_still_works(self):
        notifier = PromptChangeNotifier()
        event = PromptChangeEvent(
            prompt_key="default",
            old_version=1,
            new_version=2,
            changelog="test",
            trigger="manual",
        )
        # Should not raise
        notifier.notify(event)
