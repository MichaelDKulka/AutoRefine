"""Production analytics for AutoRefine — improvement curves, failure
patterns, refinement effectiveness, ROI reporting.

Usage::

    analytics = Analytics(store, prompt_key="default")
    snap = analytics.snapshot(days=30)
    curve = analytics.improvement_curve(days=30)
    eff = analytics.refinement_effectiveness()
    failures = analytics.common_failure_patterns(top_n=10)
    report = analytics.generate_roi_report()
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from autorefine.models import FeedbackSignal, Interaction
from autorefine.storage.base import BaseStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnalyticsSnapshot:
    """Point-in-time metrics for a prompt key."""

    prompt_key: str = "default"
    total_interactions: int = 0
    total_feedback: int = 0
    average_score: float = 0.0
    positive_rate: float = 0.0
    negative_rate: float = 0.0
    active_version: int = 0
    total_versions: int = 0
    total_cost: float = 0.0
    refiner_cost: float = 0.0
    score_by_version: dict[int, float] = field(default_factory=dict)
    feedback_distribution: dict[str, int] = field(default_factory=dict)
    improvement_curve: list[dict] = field(default_factory=list)


@dataclass
class RefinementEffectivenessEntry:
    """Score delta for a single refinement (version transition)."""

    version: int
    parent_version: int | None
    score_before: float
    score_after: float
    delta: float
    feedback_count_before: int
    feedback_count_after: int
    changelog: str


@dataclass
class FailurePattern:
    """A recurring keyword/phrase from negative feedback."""

    pattern: str
    count: int
    example_comments: list[str]


# ═══════════════════════════════════════════════════════════════════════
# Stop words for keyword extraction
# ═══════════════════════════════════════════════════════════════════════

_STOP_WORDS = frozenset(
    ["the", "a", "an", "is", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "can", "could", "must", "need", "dare", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "it", "its", "this", "that", "these", "those", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their", "what", "which", "who", "whom", "and", "but", "if", "or", "because", "until", "while", "about", "up", "just", "also", "don't", "didn't", "doesn't", "isn't", "wasn't", "aren't", "response", "answer", "question", "asked"]
)


# ═══════════════════════════════════════════════════════════════════════
# Analytics engine
# ═══════════════════════════════════════════════════════════════════════

class Analytics:
    """Computes performance metrics, improvement curves, and ROI reports."""

    def __init__(self, store: BaseStore, prompt_key: str = "default") -> None:
        self._store = store
        self._prompt_key = prompt_key

    # ── Core snapshot ────────────────────────────────────────────────

    def snapshot(self, days: int = 30) -> AnalyticsSnapshot:
        """Compute a full analytics snapshot for the last N days."""
        since = _utc_now() - timedelta(days=days)

        interactions = self._store.get_interactions(
            prompt_key=self._prompt_key, limit=100_000, since=since)
        feedback = self._store.get_feedback(
            prompt_key=self._prompt_key, limit=100_000, since=since)
        history = self._store.get_prompt_history(self._prompt_key)
        active = self._store.get_active_prompt(self._prompt_key)

        total_fb = len(feedback)
        avg_score = (sum(fb.score for fb in feedback) / total_fb) if total_fb else 0.0

        dist, positive, negative = self._compute_distribution(feedback)
        pos_rate = positive / total_fb if total_fb else 0.0
        neg_rate = negative / total_fb if total_fb else 0.0

        avg_by_version = self._compute_score_by_version(interactions, feedback)

        curve = [
            {"version": pv.version, "score": avg_by_version.get(pv.version, 0.0),
             "created_at": pv.created_at.isoformat(), "changelog": pv.changelog}
            for pv in history
        ]

        monthly_refiner = self._store.get_monthly_refiner_cost()

        return AnalyticsSnapshot(
            prompt_key=self._prompt_key,
            total_interactions=len(interactions), total_feedback=total_fb,
            average_score=avg_score, positive_rate=pos_rate, negative_rate=neg_rate,
            active_version=active.version if active else 0,
            total_versions=len(history), refiner_cost=monthly_refiner,
            score_by_version=avg_by_version, feedback_distribution=dist,
            improvement_curve=curve,
        )

    # ── Improvement curve ────────────────────────────────────────────

    def improvement_curve(self, days: int = 30) -> list[dict[str, Any]]:
        """Daily average feedback score for the prompt_key.

        Returns a list of ``{"date": "YYYY-MM-DD", "avg_score": float,
        "count": int}`` dicts, one per day with feedback.
        """
        since = _utc_now() - timedelta(days=days)
        feedback = self._store.get_feedback(
            prompt_key=self._prompt_key, limit=100_000, since=since)

        daily: dict[str, list[float]] = defaultdict(list)
        for fb in feedback:
            day = fb.created_at.strftime("%Y-%m-%d")
            daily[day].append(fb.score)

        return [
            {"date": d, "avg_score": round(sum(s) / len(s), 4), "count": len(s)}
            for d, s in sorted(daily.items())
        ]

    # ── Refinement effectiveness ─────────────────────────────────────

    def refinement_effectiveness(self) -> list[RefinementEffectivenessEntry]:
        """Score before vs after each refinement.

        For each version with a parent, compares the average feedback
        score on interactions served by the parent version vs the new
        version.
        """
        history = self._store.get_prompt_history(self._prompt_key)
        interactions = self._store.get_interactions(
            prompt_key=self._prompt_key, limit=100_000)
        feedback = self._store.get_feedback(
            prompt_key=self._prompt_key, limit=100_000)

        score_by_v = self._compute_score_by_version(interactions, feedback)
        count_by_v = self._compute_feedback_count_by_version(interactions, feedback)

        results: list[RefinementEffectivenessEntry] = []
        for pv in history:
            if pv.parent_version is None:
                continue
            before = score_by_v.get(pv.parent_version, 0.0)
            after = score_by_v.get(pv.version, 0.0)
            results.append(RefinementEffectivenessEntry(
                version=pv.version,
                parent_version=pv.parent_version,
                score_before=round(before, 4),
                score_after=round(after, 4),
                delta=round(after - before, 4),
                feedback_count_before=count_by_v.get(pv.parent_version, 0),
                feedback_count_after=count_by_v.get(pv.version, 0),
                changelog=pv.changelog,
            ))
        return results

    # ── Feedback distribution ────────────────────────────────────────

    def feedback_distribution(self, days: int = 30) -> dict[str, Any]:
        """Positive/negative/correction percentages."""
        since = _utc_now() - timedelta(days=days)
        feedback = self._store.get_feedback(
            prompt_key=self._prompt_key, limit=100_000, since=since)
        total = len(feedback)
        if total == 0:
            return {"total": 0, "breakdown": {}}

        dist, pos, neg = self._compute_distribution(feedback)
        corrections = dist.get("correction", 0)

        return {
            "total": total,
            "positive_pct": round(pos / total * 100, 1),
            "negative_pct": round(neg / total * 100, 1),
            "correction_pct": round(corrections / total * 100, 1),
            "breakdown": {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in dist.items()},
        }

    # ── Common failure patterns ──────────────────────────────────────

    def common_failure_patterns(self, days: int = 30, top_n: int = 10) -> list[FailurePattern]:
        """Extract recurring keywords from negative feedback comments.

        Groups by frequency — the most common complaint keywords rise
        to the top.  Filters stop words and short tokens.
        """
        since = _utc_now() - timedelta(days=days)
        feedback = self._store.get_feedback(
            prompt_key=self._prompt_key, limit=100_000, since=since)

        negative = [fb for fb in feedback if fb.score < 0 and fb.comment]
        if not negative:
            return []

        # Tokenize and count
        word_counts: Counter[str] = Counter()
        word_examples: dict[str, list[str]] = defaultdict(list)

        for fb in negative:
            words = set(self._extract_keywords(fb.comment))
            for w in words:
                word_counts[w] += 1
                if len(word_examples[w]) < 3:
                    word_examples[w].append(fb.comment[:120])

        # Also extract 2-grams for multi-word patterns
        for fb in negative:
            tokens = self._tokenize(fb.comment)
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                if tokens[i] not in _STOP_WORDS and tokens[i + 1] not in _STOP_WORDS:
                    word_counts[bigram] += 1
                    if len(word_examples[bigram]) < 3:
                        word_examples[bigram].append(fb.comment[:120])

        return [
            FailurePattern(pattern=word, count=count,
                           example_comments=word_examples[word])
            for word, count in word_counts.most_common(top_n)
            if count >= 2  # only patterns that appear 2+ times
        ]

    # ── Cost per improvement point ───────────────────────────────────

    def cost_per_improvement_point(self) -> float | None:
        """Total refiner cost divided by total score improvement.

        Returns ``None`` if there's been no improvement or no
        refinements.  A lower number means better ROI.
        """
        effectiveness = self.refinement_effectiveness()
        if not effectiveness:
            return None

        total_delta = sum(e.delta for e in effectiveness)
        if total_delta <= 0:
            return None

        refiner_cost = self._store.get_monthly_refiner_cost()
        if refiner_cost <= 0:
            return 0.0

        return round(refiner_cost / total_delta, 4)

    # ── ROI report ───────────────────────────────────────────────────

    def generate_roi_report(self, days: int = 30) -> str:
        """Generate a formatted ROI report for stakeholders.

        Returns a human-readable string summarising spend, improvement,
        and feedback trends — suitable for sharing with a manager.
        """
        snap = self.snapshot(days)
        effectiveness = self.refinement_effectiveness()
        distribution = self.feedback_distribution(days)
        failures = self.common_failure_patterns(days, top_n=5)
        cost_per_pt = self.cost_per_improvement_point()

        lines: list[str] = []
        lines.append(f"AutoRefine ROI Report — {self._prompt_key}")
        lines.append(f"Period: last {days} days")
        lines.append("=" * 50)

        # Spend
        lines.append(f"\nRefiner spend this month: ${snap.refiner_cost:.2f}")
        lines.append(f"Budget limit: ${0:.2f}")  # placeholder

        # Volume
        lines.append(f"\nInteractions: {snap.total_interactions:,}")
        lines.append(f"Feedback signals: {snap.total_feedback:,}")
        lines.append(f"Prompt versions: {snap.total_versions}")

        # Satisfaction
        lines.append(f"\nAverage satisfaction score: {snap.average_score:+.2f}")
        pos_pct = distribution.get("positive_pct", 0)
        neg_pct = distribution.get("negative_pct", 0)
        lines.append(f"Positive feedback: {pos_pct:.0f}%")
        lines.append(f"Negative feedback: {neg_pct:.0f}%")

        # Improvement
        if effectiveness:
            first_score = effectiveness[0].score_before
            last_score = effectiveness[-1].score_after
            total_delta = last_score - first_score

            if first_score != 0:
                pct_improvement = abs(total_delta / abs(first_score)) * 100
            else:
                pct_improvement = 0

            lines.append(f"\nScore improvement: {first_score:+.2f} -> {last_score:+.2f}")
            lines.append(f"Total improvement: {total_delta:+.2f} ({pct_improvement:.0f}%)")
            lines.append(f"Refinements applied: {len(effectiveness)}")

            if cost_per_pt is not None:
                lines.append(f"Cost per improvement point: ${cost_per_pt:.2f}")

            # The headline
            if total_delta > 0 and snap.refiner_cost > 0:
                lines.append(
                    f'\n>>> AutoRefine spent ${snap.refiner_cost:.2f} on refinement this month. '
                    f'Your average user satisfaction improved from {first_score:+.2f} to '
                    f'{last_score:+.2f}, a {pct_improvement:.0f}% improvement.'
                )
            elif total_delta > 0:
                lines.append(
                    f'\n>>> User satisfaction improved from {first_score:+.2f} to '
                    f'{last_score:+.2f} with zero refiner cost (under budget).'
                )
        else:
            lines.append("\nNo refinements applied yet.")

        # Top failure patterns
        if failures:
            lines.append("\nTop complaint patterns:")
            for fp in failures[:5]:
                lines.append(f'  - "{fp.pattern}" ({fp.count} mentions)')

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_distribution(
        feedback: list[FeedbackSignal],
    ) -> tuple[dict[str, int], int, int]:
        """Return (distribution_dict, positive_count, negative_count)."""
        dist: dict[str, int] = {}
        positive = 0
        negative = 0
        for fb in feedback:
            key = fb.feedback_type.value
            dist[key] = dist.get(key, 0) + 1
            if fb.score > 0:
                positive += 1
            elif fb.score < 0:
                negative += 1
        return dist, positive, negative

    @staticmethod
    def _compute_score_by_version(
        interactions: list[Interaction],
        feedback: list[FeedbackSignal],
    ) -> dict[int, float]:
        """Map version -> average feedback score."""
        ix_versions: dict[str, int] = {ix.id: ix.prompt_version for ix in interactions}
        version_scores: dict[int, list[float]] = defaultdict(list)
        for fb in feedback:
            v = ix_versions.get(fb.interaction_id, 0)
            version_scores[v].append(fb.score)
        return {
            v: sum(scores) / len(scores)
            for v, scores in version_scores.items()
            if scores
        }

    @staticmethod
    def _compute_feedback_count_by_version(
        interactions: list[Interaction],
        feedback: list[FeedbackSignal],
    ) -> dict[int, int]:
        """Map version -> feedback count."""
        ix_versions: dict[str, int] = {ix.id: ix.prompt_version for ix in interactions}
        counts: dict[int, int] = defaultdict(int)
        for fb in feedback:
            v = ix_versions.get(fb.interaction_id, 0)
            counts[v] += 1
        return dict(counts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase tokens, stripping punctuation."""
        return re.findall(r"[a-z']+", text.lower())

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords (no stop words, min 3 chars)."""
        tokens = re.findall(r"[a-z']+", text.lower())
        return [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]
