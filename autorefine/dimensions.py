"""Structured feedback dimensions — multi-axis quality scoring.

Developers define named quality axes (accuracy, reasoning, tone, etc.)
specific to their domain.  Each axis has a description, scale, weight, and
refinement priority.  The :class:`DimensionAggregator` computes per-dimension
statistics across a batch of feedback for the refiner.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from autorefine.models import FeedbackSignal

logger = logging.getLogger("autorefine.dimensions")

_RESERVED_NAMES = frozenset({"score", "overall", "composite", "id", "type"})
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── Dimension definition ─────────────────────────────────────────────

class FeedbackDimension(BaseModel):
    """Definition of a single feedback quality axis."""

    name: str = Field(description="Lowercase alphanumeric + underscores, max 64 chars.")
    description: str = Field(description="Human-readable, injected into meta-prompt for the refiner.")
    scale: tuple[float, float] = (-1.0, 1.0)
    weight: float = Field(default=1.0, ge=0.0)
    refinement_priority: Literal["high", "medium", "low"] = "medium"

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                f"Dimension name must be lowercase alphanumeric + underscores, "
                f"max 64 chars, got: {v!r}"
            )
        if v in _RESERVED_NAMES:
            raise ValueError(f"Dimension name {v!r} is reserved")
        return v

    @field_validator("scale")
    @classmethod
    def _validate_scale(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(f"Scale min must be < max, got {v}")
        return v


# ── Dimension schema ─────────────────────────────────────────────────

class FeedbackDimensionSchema(BaseModel):
    """Complete dimension configuration for a prompt_key."""

    prompt_key: str = "default"
    dimensions: dict[str, FeedbackDimension] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    def normalize_score(self, dimension_name: str, raw_score: float) -> float:
        """Convert a score from the dimension's native scale to [-1, 1]."""
        dim = self.dimensions.get(dimension_name)
        if dim is None:
            return max(-1.0, min(1.0, raw_score))

        lo, hi = dim.scale
        if lo == -1.0 and hi == 1.0:
            return max(-1.0, min(1.0, raw_score))

        # Clamp to dimension scale
        clamped = max(lo, min(hi, raw_score))
        # Linear map [lo, hi] → [-1, 1]
        span = hi - lo
        if span == 0:
            return 0.0
        return 2.0 * (clamped - lo) / span - 1.0

    def compute_composite(self, dimension_scores: dict[str, float]) -> float:
        """Weighted average of normalized dimension scores.

        Only dimensions that are actually scored contribute.
        Weights are re-normalized over the scored subset.
        """
        if not dimension_scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for name, score in dimension_scores.items():
            dim = self.dimensions.get(name)
            w = dim.weight if dim else 1.0
            normalized = self.normalize_score(name, score)
            weighted_sum += w * normalized
            total_weight += w

        if total_weight == 0:
            return 0.0
        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def normalized_weights(self) -> dict[str, float]:
        """Return weights that sum to 1.0."""
        total = sum(d.weight for d in self.dimensions.values())
        if total == 0:
            return {name: 0.0 for name in self.dimensions}
        return {name: d.weight / total for name, d in self.dimensions.items()}

    @classmethod
    def from_dict(
        cls, prompt_key: str, raw: dict[str, dict[str, Any]]
    ) -> FeedbackDimensionSchema:
        """Build a schema from the developer-facing config dict."""
        dims: dict[str, FeedbackDimension] = {}
        for name, spec in raw.items():
            dims[name] = FeedbackDimension(name=name, **spec)
        return cls(prompt_key=prompt_key, dimensions=dims)


# ── Aggregation results ──────────────────────────────────────────────

class DimensionSummary(BaseModel):
    """Aggregated statistics for a single dimension across a feedback batch."""

    name: str
    description: str
    mean_score: float
    signal_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    weight: float
    refinement_priority: str
    trend: str = "unknown"  # "improving", "declining", "stable", "unknown"


# ── Aggregator ───────────────────────────────────────────────────────

class DimensionAggregator:
    """Aggregates dimensional feedback across multiple signals for the refiner."""

    def __init__(self, schema: FeedbackDimensionSchema) -> None:
        self._schema = schema

    def aggregate(
        self, signals: list[FeedbackSignal]
    ) -> dict[str, DimensionSummary]:
        """Compute per-dimension statistics across a batch of feedback.

        Dimensions with fewer than 5 signals report as insufficient data
        (mean_score=0, trend="unknown").
        """
        # Collect scores per dimension
        dim_scores: dict[str, list[float]] = {
            name: [] for name in self._schema.dimensions
        }

        for sig in signals:
            for dim_name, score in sig.dimensions.items():
                if dim_name in dim_scores:
                    normalized = self._schema.normalize_score(dim_name, score)
                    dim_scores[dim_name].append(normalized)

        results: dict[str, DimensionSummary] = {}
        for name, dim in self._schema.dimensions.items():
            scores = dim_scores[name]
            count = len(scores)

            if count < 5:
                results[name] = DimensionSummary(
                    name=name,
                    description=dim.description,
                    mean_score=0.0,
                    signal_count=count,
                    positive_count=sum(1 for s in scores if s > 0),
                    negative_count=sum(1 for s in scores if s < 0),
                    neutral_count=sum(1 for s in scores if s == 0),
                    weight=dim.weight,
                    refinement_priority=dim.refinement_priority,
                    trend="unknown",
                )
                continue

            mean = sum(scores) / count
            pos = sum(1 for s in scores if s > 0)
            neg = sum(1 for s in scores if s < 0)
            neu = count - pos - neg

            # Trend: compare first half vs second half
            trend = "unknown"
            if count >= 10:
                mid = count // 2
                first_half = sum(scores[:mid]) / mid
                second_half = sum(scores[mid:]) / (count - mid)
                delta = second_half - first_half
                if delta > 0.1:
                    trend = "improving"
                elif delta < -0.1:
                    trend = "declining"
                else:
                    trend = "stable"

            results[name] = DimensionSummary(
                name=name,
                description=dim.description,
                mean_score=round(mean, 4),
                signal_count=count,
                positive_count=pos,
                negative_count=neg,
                neutral_count=neu,
                weight=dim.weight,
                refinement_priority=dim.refinement_priority,
                trend=trend,
            )

        return results

    def format_for_meta_prompt(
        self, signals: list[FeedbackSignal]
    ) -> str:
        """Render dimension analysis as a formatted block for the meta-prompt."""
        summaries = self.aggregate(signals)
        if not summaries:
            return ""

        total = len(signals)
        n_dims = len(self._schema.dimensions)

        lines = [f"DIMENSION ANALYSIS ({total} signals, {n_dims} dimensions configured):\n"]

        # Sort: high priority first, then by severity (most negative first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_dims = sorted(
            summaries.values(),
            key=lambda s: (priority_order.get(s.refinement_priority, 1), s.mean_score),
        )

        for s in sorted_dims:
            if s.signal_count < 5:
                status = "INSUFFICIENT DATA"
                icon = "?"
            elif s.mean_score < -0.2:
                status = "CRITICAL"
                icon = "X"
            elif s.mean_score < 0:
                status = "WARNING"
                icon = "!"
            elif s.mean_score > 0.5:
                status = "STRONG"
                icon = "+"
            else:
                status = "HEALTHY"
                icon = "~"

            neg_pct = (
                f"{100 * s.negative_count / s.signal_count:.0f}%"
                if s.signal_count > 0 else "N/A"
            )
            pos_pct = (
                f"{100 * s.positive_count / s.signal_count:.0f}%"
                if s.signal_count > 0 else "N/A"
            )

            lines.append(
                f"  {icon}  {s.name:<16} [{status}]  avg: {s.mean_score:+.2f}  "
                f"({s.signal_count} scored: {s.positive_count} pos / "
                f"{s.negative_count} neg / {s.neutral_count} neutral)  "
                f"weight: {s.weight}"
            )
            if s.signal_count >= 5:
                if s.mean_score < -0.2:
                    lines.append(
                        f"    -> {neg_pct} negative. This dimension needs improvement."
                    )
                elif s.mean_score > 0.5:
                    lines.append(
                        f"    -> {pos_pct} positive. Users are satisfied. Do not change."
                    )
            lines.append("")

        # Composite
        schema = self._schema
        all_dim_scores: dict[str, list[float]] = {n: [] for n in schema.dimensions}
        for sig in signals:
            for dn, sc in sig.dimensions.items():
                if dn in all_dim_scores:
                    all_dim_scores[dn].append(schema.normalize_score(dn, sc))
        dim_means = {n: (sum(s) / len(s) if s else 0.0) for n, s in all_dim_scores.items()}
        composite = schema.compute_composite(
            {n: m for n, m in dim_means.items() if all_dim_scores[n]}
        )
        lines.append(f"COMPOSITE: {composite:+.2f} (weighted by dimension importance)\n")

        # Priority order for refinement
        critical = [s for s in sorted_dims if s.mean_score < -0.2 and s.signal_count >= 5]
        healthy = [s for s in sorted_dims if s.mean_score >= 0 and s.signal_count >= 5]

        if critical or healthy:
            lines.append("PRIORITY ORDER FOR REFINEMENT:")
            rank = 1
            for s in critical:
                lines.append(
                    f"  {rank}. {s.name} ({s.refinement_priority} priority, {s.mean_score:+.2f})"
                )
                rank += 1
            for s in healthy:
                action = "do not touch" if s.mean_score > 0.5 else "monitor only"
                lines.append(f"  {rank}. {s.name} ({action})")
                rank += 1

        return "\n".join(lines)
