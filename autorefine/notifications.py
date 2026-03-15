"""Prompt-change alerting — fires webhooks and callbacks when a prompt version changes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

logger = logging.getLogger("autorefine.notifications")


class PromptChangeEvent:
    """Immutable record of a prompt version change."""

    def __init__(
        self,
        prompt_key: str,
        old_version: int,
        new_version: int,
        changelog: str,
        trigger: str,  # "auto_refine", "manual", "rollback", "ab_test_promote"
        old_prompt: str = "",
        new_prompt: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        self.prompt_key = prompt_key
        self.old_version = old_version
        self.new_version = new_version
        self.changelog = changelog
        self.trigger = trigger
        self.old_prompt = old_prompt
        self.new_prompt = new_prompt
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "prompt_change",
            "prompt_key": self.prompt_key,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "changelog": self.changelog,
            "trigger": self.trigger,
            "old_prompt_preview": self.old_prompt[:200] if self.old_prompt else "",
            "new_prompt_preview": self.new_prompt[:200] if self.new_prompt else "",
            "timestamp": self.timestamp.isoformat(),
        }


class PromptChangeNotifier:
    """Dispatches prompt-change alerts via callbacks and/or webhooks."""

    def __init__(
        self,
        webhook_url: str = "",
        on_prompt_change: Callable[[PromptChangeEvent], Any] | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._on_prompt_change = on_prompt_change

    def notify(self, event: PromptChangeEvent) -> None:
        """Fire all configured notification channels."""
        logger.info(
            "Prompt changed: key=%s v%d→v%d trigger=%s — %s",
            event.prompt_key,
            event.old_version,
            event.new_version,
            event.trigger,
            event.changelog,
        )

        if self._on_prompt_change:
            try:
                self._on_prompt_change(event)
            except Exception:
                logger.error("on_prompt_change callback failed", exc_info=True)

        if self._webhook_url:
            self._send_webhook(event)

    def _send_webhook(self, event: PromptChangeEvent) -> None:
        """POST event payload to the configured webhook URL."""
        try:
            resp = httpx.post(
                self._webhook_url,
                json=event.to_dict(),
                timeout=10.0,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Webhook returned %d: %s", resp.status_code, resp.text[:200]
                )
        except Exception:
            logger.error("Webhook delivery failed for %s", self._webhook_url, exc_info=True)
