"""Abstract base class for all AutoRefine storage backends.

Every backend must implement the full set of abstract methods defined here.
Concrete convenience methods (``rollback_to_version``, ``get_analytics``)
are provided on the base class so subclasses get them for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

from autorefine.models import (
    ABTest,
    CostEntry,
    FeedbackSignal,
    Interaction,
    PromptVersion,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BaseStore(ABC):
    """Interface that every storage backend must implement."""

    # ── Interactions ──────────────────────────────────────────────────

    @abstractmethod
    def save_interaction(self, interaction: Interaction) -> None:
        """Persist a single interaction (request/response pair)."""
        ...

    @abstractmethod
    def get_interaction(self, interaction_id: str) -> Interaction | None:
        """Retrieve an interaction by its unique ID, or ``None``."""
        ...

    @abstractmethod
    def get_interactions(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Interaction]:
        """Return interactions for *prompt_key*, newest first.

        Args:
            prompt_key: Filter to this prompt namespace.
            limit: Maximum number of results.
            since: Only return interactions created on or after this UTC
                timestamp.
        """
        ...

    # ── Feedback ─────────────────────────────────────────────────────

    @abstractmethod
    def save_feedback(self, feedback: FeedbackSignal) -> None:
        """Persist a single feedback signal."""
        ...

    @abstractmethod
    def get_feedback(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
        unprocessed_only: bool = False,
    ) -> list[FeedbackSignal]:
        """Return feedback signals for *prompt_key*, newest first.

        Args:
            prompt_key: Filter to feedback on interactions belonging to
                this prompt namespace.
            limit: Maximum number of results.
            since: Only return feedback created on or after this time.
            unprocessed_only: If ``True``, exclude feedback that has
                already been consumed by a refinement cycle.
        """
        ...

    @abstractmethod
    def mark_feedback_processed(self, feedback_ids: list[str]) -> None:
        """Mark the given feedback IDs as consumed by a refinement cycle."""
        ...

    # ── Prompt versions ──────────────────────────────────────────────

    @abstractmethod
    def save_prompt_version(self, version: PromptVersion) -> None:
        """Persist a prompt version (insert or upsert)."""
        ...

    @abstractmethod
    def get_active_prompt(self, prompt_key: str = "default") -> PromptVersion | None:
        """Return the currently active prompt version, or ``None``."""
        ...

    @abstractmethod
    def get_prompt_version(
        self, prompt_key: str, version: int
    ) -> PromptVersion | None:
        """Return a specific version by number, or ``None``."""
        ...

    @abstractmethod
    def get_prompt_history(self, prompt_key: str = "default") -> list[PromptVersion]:
        """Return all versions for *prompt_key*, ordered by version number."""
        ...

    @abstractmethod
    def set_active_version(self, prompt_key: str, version: int) -> None:
        """Make *version* the active prompt; deactivate all others in the
        same *prompt_key* namespace.

        Passing ``version=-1`` deactivates all versions (used before
        inserting a new active version).
        """
        ...

    # ── A/B tests ────────────────────────────────────────────────────

    @abstractmethod
    def save_ab_test(self, ab_test: ABTest) -> None:
        """Persist a new or updated A/B test record."""
        ...

    @abstractmethod
    def get_active_ab_test(self, prompt_key: str = "default") -> ABTest | None:
        """Return the currently active A/B test, or ``None``."""
        ...

    @abstractmethod
    def update_ab_test(self, ab_test: ABTest) -> None:
        """Update an existing A/B test record (scores, status, etc.)."""
        ...

    # ── Cost tracking ────────────────────────────────────────────────

    @abstractmethod
    def save_cost_entry(self, entry: CostEntry) -> None:
        """Persist a cost record for an LLM call."""
        ...

    @abstractmethod
    def get_monthly_refiner_cost(self) -> float:
        """Return the total refiner spend in USD for the current calendar month."""
        ...

    # ── Refinement directives ────────────────────────────────────────

    def save_refinement_directives(self, directives: Any) -> None:
        """Persist refinement directives for a prompt_key.

        Default no-op for backward compat — overridden by backends.
        """

    def get_refinement_directives(self, prompt_key: str) -> Any:
        """Retrieve refinement directives for a prompt_key, or None."""
        return None

    # ── Dimension schemas ────────────────────────────────────────────

    def save_dimension_schema(self, schema: Any) -> None:
        """Persist a feedback dimension schema for a prompt_key.

        Default no-op for backward compat — overridden by backends.
        """

    def get_dimension_schema(self, prompt_key: str) -> Any:
        """Retrieve a feedback dimension schema, or None."""
        return None

    # ── Maintenance ──────────────────────────────────────────────────

    @abstractmethod
    def purge_old_data(self, before: datetime) -> int:
        """Delete interactions, feedback, and cost entries created before
        *before*.

        Returns:
            The total number of rows/records deleted.
        """
        ...

    # ── Concrete convenience methods ─────────────────────────────────

    def rollback_to_version(self, prompt_key: str, version: int) -> PromptVersion | None:
        """Convenience: activate a previous version (rollback).

        Returns the activated :class:`PromptVersion`, or ``None`` if the
        version does not exist.
        """
        pv = self.get_prompt_version(prompt_key, version)
        if pv is None:
            return None
        self.set_active_version(prompt_key, version)
        return pv

    def get_analytics(self, prompt_key: str = "default", days: int = 30) -> dict[str, Any]:
        """Return a lightweight analytics snapshot for *prompt_key*.

        This is a convenience built on the primitive query methods so
        every backend gets it for free.  For heavier analysis, use the
        :class:`~autorefine.analytics.Analytics` class directly.

        Returns:
            A dict with keys: ``total_interactions``, ``total_feedback``,
            ``positive_rate``, ``negative_rate``, ``average_score``,
            ``active_version``, ``total_versions``,
            ``monthly_refiner_cost``.
        """
        since = _utc_now() - timedelta(days=days)

        interactions = self.get_interactions(prompt_key=prompt_key, limit=100_000, since=since)
        feedback = self.get_feedback(prompt_key=prompt_key, limit=100_000, since=since)
        history = self.get_prompt_history(prompt_key)
        active = self.get_active_prompt(prompt_key)

        total_fb = len(feedback)
        positives = sum(1 for fb in feedback if fb.score > 0)
        negatives = sum(1 for fb in feedback if fb.score < 0)
        avg_score = (sum(fb.score for fb in feedback) / total_fb) if total_fb else 0.0

        return {
            "total_interactions": len(interactions),
            "total_feedback": total_fb,
            "positive_rate": positives / total_fb if total_fb else 0.0,
            "negative_rate": negatives / total_fb if total_fb else 0.0,
            "average_score": avg_score,
            "active_version": active.version if active else 0,
            "total_versions": len(history),
            "monthly_refiner_cost": self.get_monthly_refiner_cost(),
        }
