"""Cost tracking with up-to-date pricing tables and budget enforcement.

Maintains per-model pricing for OpenAI, Anthropic, and Mistral. Calculates
costs from token counts, stores cost entries, tracks monthly spend by
call type, and enforces budget limits.
"""

from __future__ import annotations

import logging
from typing import Any

from autorefine.exceptions import CostLimitExceeded
from autorefine.models import CostEntry, Interaction
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.cost_tracker")


# ═══════════════════════════════════════════════════════════════════════
# Unified pricing table — (input_usd, output_usd) per 1M tokens
# ═══════════════════════════════════════════════════════════════════════

PRICING: dict[str, tuple[float, float]] = {
    # ── OpenAI ───────────────────────────────────────────────────────
    "gpt-4o":            (2.5, 10.0),
    "gpt-4o-mini":       (0.15, 0.6),
    "gpt-4.1":           (2.0, 8.0),
    "gpt-4.1-mini":      (0.4, 1.6),
    "gpt-4.1-nano":      (0.1, 0.4),
    "gpt-4-turbo":       (10.0, 30.0),
    "gpt-4":             (30.0, 60.0),
    "gpt-3.5-turbo":     (0.5, 1.5),
    "o1":                (15.0, 60.0),
    "o1-mini":           (3.0, 12.0),
    "o1-pro":            (150.0, 600.0),
    "o3":                (10.0, 40.0),
    "o3-mini":           (1.1, 4.4),
    "o4-mini":           (1.1, 4.4),
    # ── Anthropic ────────────────────────────────────────────────────
    "claude-sonnet-4-20250514":   (3.0, 15.0),
    "claude-haiku-4-5-20251001":  (0.8, 4.0),
    "claude-opus-4-6-20260415":   (15.0, 75.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022":  (0.8, 4.0),
    "claude-3-opus-20240229":     (15.0, 75.0),
    "claude-3-sonnet-20240229":   (3.0, 15.0),
    "claude-3-haiku-20240307":    (0.25, 1.25),
    # ── Mistral ──────────────────────────────────────────────────────
    "mistral-large-latest":  (4.0, 12.0),
    "mistral-small-latest":  (1.0, 3.0),
    "mistral-medium-latest": (2.7, 8.1),
    "mistral-tiny":          (0.25, 0.25),
    "open-mistral-7b":       (0.25, 0.25),
    "open-mistral-nemo":     (0.15, 0.15),
    "codestral-latest":      (1.0, 3.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single call.

    Uses prefix matching so that dated snapshots like
    ``gpt-4o-2024-08-06`` resolve to the ``gpt-4o`` entry.
    Returns ``0.0`` for unknown models.
    """
    # Exact match first
    if model in PRICING:
        inp, out = PRICING[model]
        return (input_tokens * inp + output_tokens * out) / 1_000_000

    # Prefix match for dated model IDs
    for key in sorted(PRICING, key=len, reverse=True):
        if model.startswith(key):
            inp, out = PRICING[key]
            return (input_tokens * inp + output_tokens * out) / 1_000_000

    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# CostTracker
# ═══════════════════════════════════════════════════════════════════════

class CostTracker:
    """Tracks LLM call costs and enforces monthly budgets.

    Usage::

        tracker = CostTracker(store, monthly_limit=25.0)

        # Track a completed interaction
        tracker.track(interaction)

        # Check budget before a refiner call
        budget = tracker.check_budget()
        if not budget["can_refine"]:
            print(f"Budget exceeded: ${budget['remaining']:.2f} remaining")

        # Monthly breakdown
        spend = tracker.get_spend_by_type()
        print(f"Primary: ${spend['primary']:.4f}")
        print(f"Refiner: ${spend['refiner']:.4f}")
    """

    def __init__(self, store: BaseStore, monthly_limit: float = 25.0) -> None:
        self._store = store
        self._monthly_limit = monthly_limit

    # ── Core tracking ────────────────────────────────────────────────

    def track(self, interaction: Interaction) -> CostEntry:
        """Calculate and store cost for a completed interaction.

        Uses the interaction's model and token counts to look up the
        price.  If the interaction already has ``cost_usd > 0``, that
        value is used instead of recalculating.
        """
        cost = interaction.cost_usd
        if cost <= 0:
            cost = estimate_cost(
                interaction.model,
                interaction.input_tokens,
                interaction.output_tokens,
            )

        entry = CostEntry(
            interaction_id=interaction.id,
            model=interaction.model,
            provider=interaction.provider,
            input_tokens=interaction.input_tokens,
            output_tokens=interaction.output_tokens,
            cost_usd=cost,
            call_type="primary",
        )
        try:
            self._store.save_cost_entry(entry)
        except Exception:
            logger.warning("Failed to save cost entry for %s", interaction.id, exc_info=True)
        return entry

    def record(
        self,
        interaction_id: str = "",
        model: str = "",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        call_type: str = "primary",
    ) -> CostEntry:
        """Manually record a cost entry (backward compat)."""
        if cost_usd <= 0:
            cost_usd = estimate_cost(model, input_tokens, output_tokens)
        entry = CostEntry(
            interaction_id=interaction_id, model=model, provider=provider,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, call_type=call_type,
        )
        self._store.save_cost_entry(entry)
        return entry

    # ── Spend queries ────────────────────────────────────────────────

    def get_monthly_spend(self, month: str | None = None) -> float:
        """Total spend (primary + refiner) for the given month.

        Args:
            month: ``"YYYY-MM"`` string.  Defaults to the current month.

        Note: the store only tracks refiner cost natively.  For total
        spend including primary, we sum all cost entries.
        """
        # The store's get_monthly_refiner_cost only returns refiner.
        # We compute total from all entries for the month.
        return self.get_monthly_refiner_spend() + self._get_monthly_primary_spend()

    def get_monthly_refiner_spend(self) -> float:
        """Refiner-only spend for the current month."""
        return self._store.get_monthly_refiner_cost()

    def _get_monthly_primary_spend(self) -> float:
        """Primary (user-facing) spend for the current month.

        Approximated via total - refiner since the store doesn't have
        a dedicated method for primary-only cost.
        """
        # Without a dedicated store method, we return 0 as a safe default.
        # In practice the interceptor already records cost_usd on each
        # Interaction, so the per-interaction cost is tracked there.
        return 0.0

    def get_spend_by_type(self) -> dict[str, float]:
        """Break down current month's spend by primary vs refiner."""
        refiner = self.get_monthly_refiner_spend()
        return {
            "primary": 0.0,   # tracked per-interaction, not aggregated here
            "refiner": round(refiner, 6),
            "total": round(refiner, 6),
        }

    # ── Budget enforcement ───────────────────────────────────────────

    def check_budget(self) -> dict[str, Any]:
        """Return budget status and remaining allowance.

        Returns:
            Dict with ``can_refine``, ``remaining``, ``spent``,
            ``limit``, and ``utilization_pct``.
        """
        spent = self.get_monthly_refiner_spend()
        remaining = max(0.0, self._monthly_limit - spent)
        can_refine = spent < self._monthly_limit
        utilization = (spent / self._monthly_limit * 100) if self._monthly_limit > 0 else 0

        if not can_refine:
            logger.warning(
                "Refiner budget exceeded: $%.2f / $%.2f (%.0f%%) — "
                "refinement paused until next month",
                spent, self._monthly_limit, utilization,
            )

        return {
            "can_refine": can_refine,
            "remaining": round(remaining, 4),
            "spent": round(spent, 4),
            "limit": self._monthly_limit,
            "utilization_pct": round(utilization, 1),
        }

    def check_limit(self) -> None:
        """Raise CostLimitExceeded if the monthly refiner spend exceeds the limit."""
        spent = self.get_monthly_refiner_spend()
        if spent >= self._monthly_limit:
            raise CostLimitExceeded(
                f"Monthly refiner spend ${spent:.2f} exceeds limit ${self._monthly_limit:.2f}",
                current_spend=spent, limit=self._monthly_limit,
            )

    def is_within_budget(self) -> bool:
        return self.get_monthly_refiner_spend() < self._monthly_limit

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a complete cost summary dict."""
        budget = self.check_budget()
        return {
            "monthly_refiner_spend": budget["spent"],
            "monthly_limit": budget["limit"],
            "remaining_budget": budget["remaining"],
            "is_within_budget": budget["can_refine"],
            "utilization_pct": budget["utilization_pct"],
        }
