"""Widget feedback endpoint — handles POSTs from the embeddable widget.

Receives JSON with ``interaction_id``, ``signal``, and optional
``comment``, then feeds it into the FeedbackCollector.
"""

from __future__ import annotations

import logging
from typing import Any

from autorefine.feedback import FeedbackCollector
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.dashboard.widget")


class WidgetFeedbackHandler:
    """Processes incoming feedback from the HTML widget."""

    def __init__(self, store: BaseStore, prompt_key: str = "default") -> None:
        self._collector = FeedbackCollector(store=store, prompt_key=prompt_key)

    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process a widget feedback POST.

        Args:
            data: JSON body with keys ``interaction_id``, ``signal``,
                and optionally ``comment``.

        Returns:
            A status dict for the JSON response.
        """
        interaction_id = data.get("interaction_id", "")
        signal = data.get("signal", "")
        comment = data.get("comment", "")

        if not interaction_id or not signal:
            return {"status": "error", "message": "interaction_id and signal are required"}

        try:
            fb = self._collector.submit(
                interaction_id=interaction_id,
                signal=signal,
                comment=comment,
            )
            logger.info(
                "Widget feedback received: %s for %s (score=%.1f)",
                signal, interaction_id[:8], fb.score,
            )
            return {
                "status": "ok",
                "feedback_id": fb.id,
                "score": fb.score,
            }
        except Exception as exc:
            logger.error("Widget feedback failed: %s", exc, exc_info=True)
            return {"status": "error", "message": str(exc)}
