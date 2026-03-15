"""Feedback noise filtering — removes unreliable signals before they reach the refiner."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import timedelta

from autorefine.models import FeedbackSignal, Interaction

logger = logging.getLogger("autorefine.feedback_filter")


class FeedbackFilter:
    """Filters out noisy or unreliable feedback before refinement.

    Catches common noise patterns:
    - Rage-clicking: same user spamming negative feedback in rapid succession
    - Contradictory signals: user gives both positive and negative on the same interaction
    - Low-context negatives: negative feedback with no comment on interactions the model
      handled correctly (high positive rate across other users)
    - Outlier users: single user responsible for a disproportionate share of negative feedback

    Usage::

        filt = FeedbackFilter()
        clean = filt.filter(feedback_items, interactions)
    """

    def __init__(
        self,
        enabled: bool = True,
        # A user submitting more than this many negative signals in the burst
        # window is flagged as rage-clicking
        rage_click_threshold: int = 5,
        rage_click_window_minutes: int = 2,
        # If a single user accounts for more than this fraction of all negative
        # feedback in the batch, their feedback is down-weighted
        outlier_user_fraction: float = 0.5,
        # Minimum feedback items to apply outlier detection (skip when batch is tiny)
        outlier_min_batch: int = 6,
    ) -> None:
        self._enabled = enabled
        self._rage_threshold = rage_click_threshold
        self._rage_window = timedelta(minutes=rage_click_window_minutes)
        self._outlier_fraction = outlier_user_fraction
        self._outlier_min_batch = outlier_min_batch

    def filter(
        self,
        feedback_items: list[FeedbackSignal],
        interactions: list[Interaction] | None = None,
    ) -> list[FeedbackSignal]:
        """Return a cleaned list with noisy signals removed or down-weighted."""
        if not self._enabled or not feedback_items:
            return feedback_items

        result = list(feedback_items)
        removed_ids: set[str] = set()

        # 1. Remove contradictory signals on the same interaction
        result, contradictions = self._remove_contradictions(result)
        removed_ids |= contradictions

        # 2. Detect rage-clicking
        result, rage = self._remove_rage_clicks(result)
        removed_ids |= rage

        # 3. Down-weight outlier users
        result = self._downweight_outlier_users(result)

        if removed_ids:
            logger.info(
                "Feedback filter removed %d noisy signal(s) from batch of %d",
                len(removed_ids),
                len(feedback_items),
            )

        return result

    # ------------------------------------------------------------------
    # Individual filter passes
    # ------------------------------------------------------------------

    def _remove_contradictions(
        self, items: list[FeedbackSignal]
    ) -> tuple[list[FeedbackSignal], set[str]]:
        """If the same user gave both positive and negative feedback on the same
        interaction, keep only the most recent one."""
        # Group by (user_id, interaction_id)
        groups: dict[tuple[str, str], list[FeedbackSignal]] = defaultdict(list)
        for fb in items:
            key = (fb.user_id or fb.id, fb.interaction_id)
            groups[key].append(fb)

        removed: set[str] = set()
        kept: list[FeedbackSignal] = []
        for _key, group in groups.items():
            if len(group) <= 1:
                kept.extend(group)
                continue

            has_pos = any(fb.score > 0 for fb in group)
            has_neg = any(fb.score < 0 for fb in group)
            if has_pos and has_neg:
                # Contradictory — keep only the latest
                latest = max(group, key=lambda fb: fb.created_at)
                kept.append(latest)
                removed |= {fb.id for fb in group if fb.id != latest.id}
            else:
                kept.extend(group)

        return kept, removed

    def _remove_rage_clicks(
        self, items: list[FeedbackSignal]
    ) -> tuple[list[FeedbackSignal], set[str]]:
        """Detect users who submit many negative signals in a short burst."""
        # Group negative feedback by user
        by_user: dict[str, list[FeedbackSignal]] = defaultdict(list)
        for fb in items:
            if fb.score < 0 and fb.user_id:
                by_user[fb.user_id].append(fb)

        rage_users: set[str] = set()
        for user_id, user_fbs in by_user.items():
            if len(user_fbs) < self._rage_threshold:
                continue
            # Sort by time and check for bursts
            sorted_fbs = sorted(user_fbs, key=lambda fb: fb.created_at)
            for i in range(len(sorted_fbs) - self._rage_threshold + 1):
                window_start = sorted_fbs[i].created_at
                window_end = sorted_fbs[i + self._rage_threshold - 1].created_at
                if window_end - window_start <= self._rage_window:
                    rage_users.add(user_id)
                    logger.debug(
                        "Rage-click detected: user=%s submitted %d negatives in %s",
                        user_id,
                        self._rage_threshold,
                        window_end - window_start,
                    )
                    break

        if not rage_users:
            return items, set()

        # Keep only one negative per rage-user (the one with the longest comment)
        removed: set[str] = set()
        kept: list[FeedbackSignal] = []
        for fb in items:
            if fb.user_id in rage_users and fb.score < 0:
                removed.add(fb.id)
            else:
                kept.append(fb)

        # Re-add the single most informative negative per rage-user
        for user_id in rage_users:
            user_negs = [fb for fb in items if fb.user_id == user_id and fb.score < 0]
            if user_negs:
                best = max(user_negs, key=lambda fb: len(fb.comment or ""))
                kept.append(best)
                removed.discard(best.id)

        return kept, removed

    def _downweight_outlier_users(
        self, items: list[FeedbackSignal]
    ) -> list[FeedbackSignal]:
        """If one user dominates negative feedback, halve their confidence scores."""
        if len(items) < self._outlier_min_batch:
            return items

        negatives = [fb for fb in items if fb.score < 0 and fb.user_id]
        if not negatives:
            return items

        counts: Counter[str] = Counter(fb.user_id for fb in negatives)
        total_neg = len(negatives)

        outlier_users: set[str] = set()
        for user_id, count in counts.items():
            if count / total_neg > self._outlier_fraction:
                outlier_users.add(user_id)
                logger.debug(
                    "Outlier user detected: user=%s has %d/%d negative signals (%.0f%%)",
                    user_id, count, total_neg, 100 * count / total_neg,
                )

        if not outlier_users:
            return items

        result: list[FeedbackSignal] = []
        for fb in items:
            if fb.user_id in outlier_users and fb.score < 0:
                # Down-weight rather than remove — they might be right
                fb_copy = fb.model_copy(update={"confidence": fb.confidence * 0.5})
                result.append(fb_copy)
            else:
                result.append(fb)

        return result
