"""Feedback collection, normalization, batching, and refinement preparation.

The :class:`FeedbackCollector` is the single entry point for all feedback
flowing into AutoRefine.  It handles:

1. **Recording** — normalises raw signals (thumbs up/down, corrections,
   custom scores) into :class:`~autorefine.models.FeedbackSignal` objects.
2. **Batching** — accumulates signals in memory and flushes to the store
   in configurable-size batches for write efficiency.
3. **Refinement gating** — answers "is there enough unprocessed feedback
   to justify a refinement cycle?" via :meth:`should_trigger_refinement`.
4. **Refinement preparation** — bundles interactions with their feedback
   into :class:`FeedbackBundle` objects ready for the refiner.
5. **Lifecycle** — marks feedback as "processed" after the refiner
   consumes it so it is never reused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from autorefine.models import (
    FeedbackSignal,
    FeedbackType,
    Interaction,
)
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.feedback")


# ── Score normalization ──────────────────────────────────────────────

# Canonical score map: signal name → normalised score in [-1, 1]
_SIGNAL_SCORES: dict[str, float] = {
    # Explicit signals
    "thumbs_up": 1.0,
    "positive": 1.0,
    "thumbs_down": -1.0,
    "negative": -1.0,
    "correction": -0.5,
    # Implicit signals (lower magnitude — weaker evidence)
    "implicit_reask": -0.4,
    "implicit_abandon": -0.3,
}

# Confidence weights by feedback type
_CONFIDENCE: dict[FeedbackType, float] = {
    FeedbackType.POSITIVE: 1.0,
    FeedbackType.NEGATIVE: 1.0,
    FeedbackType.CORRECTION: 0.9,
    FeedbackType.IMPLICIT_REASK: 0.5,
    FeedbackType.IMPLICIT_ABANDON: 0.3,
}

# Map friendly signal names → FeedbackType enum values
_SIGNAL_TO_TYPE: dict[str, FeedbackType] = {
    "thumbs_up": FeedbackType.POSITIVE,
    "positive": FeedbackType.POSITIVE,
    "thumbs_down": FeedbackType.NEGATIVE,
    "negative": FeedbackType.NEGATIVE,
    "correction": FeedbackType.CORRECTION,
    "implicit_reask": FeedbackType.IMPLICIT_REASK,
    "implicit_abandon": FeedbackType.IMPLICIT_ABANDON,
}


def normalise_score(
    feedback_type: FeedbackType | str,
    raw_score: float | None = None,
) -> float:
    """Convert a feedback type + optional raw score into a normalised [-1, 1] score.

    If *raw_score* is provided, it is clamped to [-1, 1] and returned
    directly (the caller knows best).  Otherwise the canonical score for
    the *feedback_type* is used.
    """
    if raw_score is not None:
        return max(-1.0, min(1.0, raw_score))

    key = feedback_type.value if isinstance(feedback_type, FeedbackType) else feedback_type
    return _SIGNAL_SCORES.get(key, 0.0)


def confidence_for_type(feedback_type: FeedbackType) -> float:
    """Return the confidence weight for a given feedback type."""
    return _CONFIDENCE.get(feedback_type, 0.5)


def _resolve_feedback_type(signal: str | FeedbackType) -> FeedbackType:
    """Convert a string signal name or FeedbackType into a FeedbackType."""
    if isinstance(signal, FeedbackType):
        return signal
    # Try the friendly-name map first, then the enum value directly
    if signal in _SIGNAL_TO_TYPE:
        return _SIGNAL_TO_TYPE[signal]
    return FeedbackType(signal)


# ── FeedbackBundle — interaction + its feedback, ready for the refiner

@dataclass
class FeedbackBundle:
    """An interaction paired with all its associated feedback signals.

    This is the unit of data the refiner consumes: it needs to see what
    the user asked, what the model answered, and how users reacted.
    """

    interaction: Interaction
    """The LLM request/response pair."""

    feedback: list[FeedbackSignal] = field(default_factory=list)
    """All feedback signals attached to this interaction."""


# ── The collector ────────────────────────────────────────────────────

class FeedbackCollector:
    """Manages feedback ingestion, in-memory batching, and refinement prep.

    Args:
        store: Storage backend for persisting feedback and querying
            interactions.
        prompt_key: Default prompt namespace for this collector.
        refine_threshold: Minimum unprocessed feedback count before
            :meth:`should_trigger_refinement` returns ``True`` and the
            ``on_ready`` callback fires.
        batch_size: Maximum signals held in memory before auto-flushing
            to the store.  Set to ``1`` for immediate persistence (the
            default for backward compat).
        on_ready: Optional callback invoked when the unprocessed count
            reaches ``refine_threshold``.  Typically wired to the
            client's ``_run_refinement`` method.

    Usage::

        collector = FeedbackCollector(store, prompt_key="support")

        # Explicit feedback via record()
        collector.record("ix-abc123", "thumbs_up")
        collector.record("ix-abc123", "thumbs_down", comment="Too verbose")
        collector.record("ix-abc123", "correction", comment="Better answer here")
        collector.record("ix-abc123", "positive", score=0.8)  # custom score

        # Check if we have enough for refinement
        if collector.should_trigger_refinement("support"):
            bundles = collector.get_refinement_batch("support", limit=50)
            # ... send bundles to refiner ...
            collector.mark_batch_processed(bundles)
    """

    def __init__(
        self,
        store: BaseStore,
        prompt_key: str = "default",
        refine_threshold: int = 20,
        batch_size: int = 1,
        on_ready: Callable[[], Any] | None = None,
    ) -> None:
        self._store = store
        self._prompt_key = prompt_key
        self._refine_threshold = refine_threshold
        self._batch_size = max(1, batch_size)
        self._on_ready = on_ready

        # In-memory batch buffer
        self._buffer: list[FeedbackSignal] = []

    # ── Primary recording API ────────────────────────────────────────

    def record(
        self,
        interaction_id: str,
        signal_type: str | FeedbackType,
        score: float | None = None,
        comment: str | None = None,
        user_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackSignal:
        """Record an explicit feedback signal for an interaction.

        Signal normalization:

        ========== ===== ===================================
        signal     score notes
        ========== ===== ===================================
        thumbs_up   1.0  alias for ``positive``
        thumbs_down -1.0 alias for ``negative``
        correction -0.5  *comment* is stored as the
                         correction text in metadata
        positive    1.0
        negative   -1.0
        (custom)   pass  *score* is clamped to [-1, 1]
        ========== ===== ===================================

        Args:
            interaction_id: The interaction this feedback refers to.
            signal_type: Signal name — ``"thumbs_up"``, ``"thumbs_down"``,
                ``"correction"``, ``"positive"``, ``"negative"``, or a
                :class:`FeedbackType` enum value.
            score: Optional explicit score override.  When provided, the
                canonical mapping above is ignored and this value (clamped
                to [-1, 1]) is used instead.
            comment: Free-text comment.  For ``"correction"`` signals
                this is the user's preferred response text.
            user_id: Optional end-user identifier.
            metadata: Arbitrary key-value pairs.

        Returns:
            The persisted (or buffered) :class:`FeedbackSignal`.
        """
        fb_type = _resolve_feedback_type(signal_type)
        normalised = normalise_score(fb_type, score)

        # For corrections, store the comment as the correction field
        correction_text = ""
        comment_text = comment or ""
        if fb_type == FeedbackType.CORRECTION and comment_text:
            correction_text = comment_text

        fb_metadata = dict(metadata) if metadata else {}

        fb = FeedbackSignal(
            interaction_id=interaction_id,
            feedback_type=fb_type,
            score=normalised,
            confidence=confidence_for_type(fb_type),
            comment=comment_text,
            correction=correction_text,
            user_id=user_id,
            metadata=fb_metadata,
        )

        self._buffer.append(fb)
        logger.debug(
            "Feedback %s buffered for interaction %s (score=%.2f, buffer=%d/%d)",
            fb.feedback_type.value, interaction_id, fb.score,
            len(self._buffer), self._batch_size,
        )

        if len(self._buffer) >= self._batch_size:
            self.flush()

        return fb

    # ── Backward-compatible submit() — used by client.py ─────────────

    def submit(
        self,
        interaction_id: str,
        signal: str | FeedbackType,
        comment: str = "",
        correction: str = "",
        score: float | None = None,
        user_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackSignal:
        """Record a feedback signal (backward-compatible with client.py).

        Delegates to :meth:`record` but also accepts a separate
        ``correction`` parameter for :attr:`FeedbackType.CORRECTION`
        signals.
        """
        fb_type = _resolve_feedback_type(signal)
        normalised = normalise_score(fb_type, score)

        fb = FeedbackSignal(
            interaction_id=interaction_id,
            feedback_type=fb_type,
            score=normalised,
            confidence=confidence_for_type(fb_type),
            comment=comment,
            correction=correction,
            user_id=user_id,
            metadata=metadata or {},
        )

        self._buffer.append(fb)

        if len(self._buffer) >= self._batch_size:
            self.flush()

        logger.debug(
            "Feedback %s recorded for interaction %s (score=%.2f)",
            fb.feedback_type.value, interaction_id, fb.score,
        )

        self._maybe_trigger()
        return fb

    # ── Batch management ─────────────────────────────────────────────

    def flush(self) -> int:
        """Flush the in-memory buffer to the store.

        Returns:
            The number of signals flushed.
        """
        if not self._buffer:
            return 0

        count = len(self._buffer)
        for fb in self._buffer:
            try:
                self._store.save_feedback(fb)
            except Exception:
                logger.warning(
                    "Failed to save feedback %s — signal lost",
                    fb.id, exc_info=True,
                )
        self._buffer.clear()
        logger.debug("Flushed %d feedback signals to store", count)
        return count

    @property
    def buffer_size(self) -> int:
        """Number of signals currently in the in-memory buffer."""
        return len(self._buffer)

    # ── Refinement gating ────────────────────────────────────────────

    def should_trigger_refinement(self, prompt_key: str = "") -> bool:
        """Check whether enough unprocessed feedback has accumulated.

        Returns ``True`` when the unprocessed feedback count for
        *prompt_key* meets or exceeds ``refine_threshold``.
        """
        key = prompt_key or self._prompt_key

        # Flush first so the count is accurate
        self.flush()

        unprocessed = self._store.get_feedback(
            prompt_key=key,
            unprocessed_only=True,
            limit=self._refine_threshold + 1,
        )
        return len(unprocessed) >= self._refine_threshold

    def get_unprocessed_count(self, prompt_key: str = "") -> int:
        """Return the number of unprocessed feedback signals in the store.

        Does NOT include signals still in the in-memory buffer (call
        :meth:`flush` first if you need an exact count).
        """
        key = prompt_key or self._prompt_key
        return len(
            self._store.get_feedback(
                prompt_key=key,
                unprocessed_only=True,
                limit=100_000,
            )
        )

    # ── Refinement batch preparation ─────────────────────────────────

    def get_refinement_batch(
        self,
        prompt_key: str = "",
        limit: int = 50,
    ) -> list[FeedbackBundle]:
        """Pull the N most recent interactions with their feedback, bundled for the refiner.

        Each :class:`FeedbackBundle` pairs an :class:`Interaction` with
        all its associated :class:`FeedbackSignal` objects.  Only
        interactions that have unprocessed feedback are included.

        Args:
            prompt_key: Prompt namespace to query.
            limit: Maximum feedback signals to include.

        Returns:
            A list of bundles, one per interaction that has unprocessed
            feedback.
        """
        key = prompt_key or self._prompt_key

        # Ensure buffer is flushed so we get a complete picture
        self.flush()

        # Get unprocessed feedback
        feedback_items = self._store.get_feedback(
            prompt_key=key,
            unprocessed_only=True,
            limit=limit,
        )

        if not feedback_items:
            return []

        # Group feedback by interaction_id
        fb_by_ix: dict[str, list[FeedbackSignal]] = {}
        for fb in feedback_items:
            fb_by_ix.setdefault(fb.interaction_id, []).append(fb)

        # Build bundles by looking up each interaction
        bundles: list[FeedbackBundle] = []
        for ix_id, fb_list in fb_by_ix.items():
            ix = self._store.get_interaction(ix_id)
            if ix is None:
                logger.warning(
                    "Interaction %s referenced by feedback but not found — skipping",
                    ix_id,
                )
                continue
            bundles.append(FeedbackBundle(interaction=ix, feedback=fb_list))

        return bundles

    def mark_batch_processed(self, bundles: list[FeedbackBundle]) -> None:
        """Mark all feedback in the given bundles as processed.

        Call this after the refiner has consumed the batch so the same
        feedback is never sent to the refiner twice.
        """
        feedback_ids: list[str] = []
        for bundle in bundles:
            feedback_ids.extend(fb.id for fb in bundle.feedback)

        if feedback_ids:
            self._store.mark_feedback_processed(feedback_ids)
            logger.info(
                "Marked %d feedback signals as processed", len(feedback_ids)
            )

    # ── Internal ─────────────────────────────────────────────────────

    def _maybe_trigger(self) -> None:
        """Check if enough unprocessed feedback has accumulated to fire on_ready."""
        if self._on_ready is None:
            return

        if self.should_trigger_refinement():
            logger.info(
                "Feedback threshold reached (%d) — triggering refinement",
                self._refine_threshold,
            )
            try:
                self._on_ready()
            except Exception:
                logger.error("Refinement callback failed", exc_info=True)

    # ── Implicit feedback detection stub ─────────────────────────────
    # TODO: Implement implicit feedback detection in a future release.
    #
    # Planned signals:
    # - IMPLICIT_REASK: user sends a semantically similar message within
    #   N seconds of the previous response → weak negative signal.
    # - IMPLICIT_ABANDON: user stops interacting within M seconds of a
    #   response without any follow-up → very weak negative signal.
    #
    # Implementation plan:
    # 1. Add detect_implicit_feedback(interaction_id, next_message, elapsed_seconds)
    #    that compares the new message to the previous one (cosine similarity
    #    via embeddings or simple token overlap).
    # 2. Add a background checker / hook that the interceptor calls after
    #    each response to check the *previous* interaction for abandonment.
    # 3. Wire into the interceptor so it's fully automatic.
    #
    # For now, callers can manually record implicit signals:
    #   collector.record(interaction_id, "implicit_reask")
    #   collector.record(interaction_id, "implicit_abandon")

    def detect_implicit_feedback(
        self,
        interaction_id: str,
        next_message: str | None = None,
        elapsed_seconds: float = 0.0,
    ) -> FeedbackSignal | None:
        """Detect implicit feedback signals from user behavior.

        .. warning::
            **NOT YET IMPLEMENTED.** This is a placeholder for future
            implicit signal detection (re-asks, abandonment).  Currently
            returns ``None`` unconditionally.

        Args:
            interaction_id: The interaction to evaluate.
            next_message: The user's follow-up message (if any).
            elapsed_seconds: Seconds since the model responded.

        Returns:
            A :class:`FeedbackSignal` if an implicit signal was
            detected, otherwise ``None``.
        """
        # ── STUB — not yet implemented ──
        # See TODO above for the implementation plan.
        return None

    # ── Async variants ───────────────────────────────────────────────

    async def async_record(
        self, interaction_id: str, signal_type: str | FeedbackType,
        score: float | None = None, comment: str | None = None,
        user_id: str = "", metadata: dict[str, Any] | None = None,
    ) -> FeedbackSignal:
        """Async version of record(). Flushes via asyncio.to_thread."""
        fb = self.record(interaction_id, signal_type, score, comment, user_id, metadata)
        return fb

    async def async_submit(
        self, interaction_id: str, signal: str | FeedbackType,
        comment: str = "", correction: str = "", score: float | None = None,
        user_id: str = "", metadata: dict[str, Any] | None = None,
    ) -> FeedbackSignal:
        """Async version of submit()."""
        fb = self.submit(interaction_id, signal, comment, correction, score, user_id, metadata)
        return fb

    async def async_flush(self) -> int:
        """Async version of flush(). Runs store writes in a thread."""
        import asyncio
        return await asyncio.to_thread(self.flush)
