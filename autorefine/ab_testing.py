"""A/B testing for validating prompt candidates before deployment.

The :class:`ABTester` splits live traffic between a **champion** (the
current active prompt) and a **candidate** (the proposed improvement).
After both variants accumulate enough interactions, a two-sample
Welch's t-test determines whether the candidate is statistically
significantly better, worse, or indistinguishable.

Key design decisions:

- **No scipy dependency.** The t-test is implemented from scratch using
  the standard library ``math`` module so AutoRefine stays lightweight.
- **Conservative by default.** The test requires ``min_interactions``
  samples per variant before it will auto-resolve.  The default
  significance level is α = 0.05.
- **Manual overrides.** :meth:`force_promote` and :meth:`force_reject`
  let dashboard users or operators bypass the statistical test.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime, timezone
from typing import Any

from autorefine.models import ABTest, PromptCandidate, PromptVersion
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.ab_testing")

# Variant labels
CHAMPION = "champion"
CANDIDATE = "candidate"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════
# Lightweight Welch's t-test (no scipy required)
# ═══════════════════════════════════════════════════════════════════════

def _welch_t_statistic(
    mean_a: float, var_a: float, n_a: int,
    mean_b: float, var_b: float, n_b: int,
) -> tuple[float, float]:
    """Compute Welch's t-statistic and approximate degrees of freedom.

    Returns:
        ``(t_stat, df)`` — the t-statistic and Welch–Satterthwaite
        degrees of freedom.
    """
    if n_a < 2 or n_b < 2:
        return 0.0, 0.0

    se_a = var_a / n_a
    se_b = var_b / n_b
    se_sum = se_a + se_b

    if se_sum == 0:
        return 0.0, 0.0

    t_stat = (mean_a - mean_b) / math.sqrt(se_sum)

    # Welch–Satterthwaite degrees of freedom
    numerator = se_sum ** 2
    denominator = (se_a ** 2 / (n_a - 1)) + (se_b ** 2 / (n_b - 1))
    if denominator == 0:
        return t_stat, 0.0
    df = numerator / denominator

    return t_stat, df


def _t_cdf_approx(t: float, df: float) -> float:
    """Approximate the CDF of the t-distribution at *t* with *df* degrees of freedom.

    Uses the regularised incomplete beta function approximation. For
    df > 100, falls back to the normal approximation (which is excellent
    for large df).

    This is accurate to ~3 decimal places for df ≥ 2, which is more
    than sufficient for A/B test decisions.
    """
    if df <= 0:
        return 0.5

    # For large df, use normal approximation
    if df > 100:
        # Φ(t) via the error function
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    # Regularised incomplete beta function approach
    x = df / (df + t * t)
    # Use a simple series expansion of the beta CDF
    # I_x(a, b) for a = df/2, b = 0.5
    a = df / 2.0
    b = 0.5

    # Compute via continued fraction (Lentz's method)
    beta_cdf = _beta_inc(a, b, x)

    if t >= 0:
        return 1.0 - 0.5 * beta_cdf
    else:
        return 0.5 * beta_cdf


def _beta_inc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction.

    Uses the Lentz algorithm, converging in ~20 iterations for typical
    A/B test parameters.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the symmetry relation if x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _beta_inc(b, a, 1.0 - x)

    # Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    # Continued fraction (Lentz's method)
    tiny = 1e-30
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, 200):
        m2 = 2 * m
        # Even step
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= c * d

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-8:
            break

    return front * (f - 1.0)


def welch_ttest_p(
    mean_a: float, var_a: float, n_a: int,
    mean_b: float, var_b: float, n_b: int,
) -> float:
    """Two-sided p-value for Welch's t-test (no scipy required).

    Tests H₀: mean_a = mean_b against H₁: mean_a ≠ mean_b.

    Args:
        mean_a, var_a, n_a: Sample mean, variance, and count for group A.
        mean_b, var_b, n_b: Same for group B.

    Returns:
        Two-sided p-value in [0, 1].
    """
    t_stat, df = _welch_t_statistic(mean_a, var_a, n_a, mean_b, var_b, n_b)
    if df < 1:
        return 1.0  # not enough data

    # Two-sided p-value
    cdf = _t_cdf_approx(abs(t_stat), df)
    p_value = 2.0 * (1.0 - cdf)
    return max(0.0, min(1.0, p_value))


# ═══════════════════════════════════════════════════════════════════════
# ABTester
# ═══════════════════════════════════════════════════════════════════════

class ABTestManager:
    """Manages A/B tests between champion and candidate prompts.

    Args:
        store: Storage backend.
        prompt_key: Default prompt namespace.
        split_ratio: Fraction of traffic routed to the candidate (0–1).
        min_interactions: Minimum samples per variant before auto-resolution.
        significance_level: α threshold for the t-test (default 0.05).

    Usage::

        tester = ABTestManager(store, prompt_key="support", split_ratio=0.2)

        # Start a test
        tester.start_test(candidate)

        # On each request, get the prompt to use
        prompt_text, variant_label = tester.get_prompt_for_request()

        # Record feedback
        tester.record_result("support", "candidate", 0.8)

        # Check if we have a winner
        winner = tester.check_significance()
        if winner == "candidate":
            tester.promote_candidate()
    """

    def __init__(
        self,
        store: BaseStore,
        prompt_key: str = "default",
        split_ratio: float = 0.2,
        min_interactions: int = 100,
        significance_level: float = 0.05,
        dimension_schema: Any = None,
    ) -> None:
        self._store = store
        self._prompt_key = prompt_key
        self._split_ratio = split_ratio
        self._min_interactions = min_interactions
        self._alpha = significance_level
        self._dimension_schema = dimension_schema

    # ── Starting a test ──────────────────────────────────────────────

    def start_test(
        self,
        candidate: PromptCandidate,
        prompt_key: str = "",
    ) -> ABTest:
        """Create a new A/B test between the current champion and a candidate.

        If an active test already exists for this prompt_key, it is
        superseded (marked as completed with result ``"superseded"``).

        Args:
            candidate: The proposed prompt to test.
            prompt_key: Override the default prompt namespace.

        Returns:
            The newly created :class:`ABTest` record.
        """
        key = prompt_key or self._prompt_key

        # Supersede any existing active test
        existing = self._store.get_active_ab_test(key)
        if existing:
            existing.is_active = False
            existing.completed_at = _utc_now()
            existing.result = "superseded"
            self._store.update_ab_test(existing)
            logger.info(
                "Superseded existing A/B test %s for key=%s",
                existing.id[:8], key,
            )

        # Save candidate as an inactive prompt version
        history = self._store.get_prompt_history(key)
        next_version = max((pv.version for pv in history), default=0) + 1

        candidate_pv = PromptVersion(
            version=next_version,
            prompt_key=key,
            system_prompt=candidate.system_prompt,
            parent_version=candidate.parent_version,
            changelog=candidate.changelog,
            is_active=False,
        )
        self._store.save_prompt_version(candidate_pv)

        # Determine champion version
        active = self._store.get_active_prompt(key)
        champion_version = active.version if active else 0

        # Create the test record
        ab_test = ABTest(
            prompt_key=key,
            control_version=champion_version,
            candidate_version=next_version,
            split_ratio=self._split_ratio,
        )
        self._store.save_ab_test(ab_test)

        logger.info(
            "Started A/B test: champion=v%d vs candidate=v%d "
            "(%.0f%% candidate traffic, min %d interactions each)",
            champion_version, next_version,
            self._split_ratio * 100, self._min_interactions,
        )
        return ab_test

    # ── Traffic routing ──────────────────────────────────────────────

    def get_prompt_for_request(
        self, prompt_key: str = "",
    ) -> tuple[str, str]:
        """Return ``(prompt_text, variant_label)`` for the current request.

        Uses ``random.random()`` against the split ratio to assign each
        request to either ``"champion"`` or ``"candidate"``.  If no test
        is active, returns the champion prompt with label ``"champion"``.

        Args:
            prompt_key: Override the default prompt namespace.

        Returns:
            A tuple of ``(prompt_text, variant_label)`` where
            ``variant_label`` is ``"champion"`` or ``"candidate"``.
        """
        key = prompt_key or self._prompt_key
        ab_test = self._store.get_active_ab_test(key)

        if ab_test is None:
            active = self._store.get_active_prompt(key)
            if active:
                return active.system_prompt, CHAMPION
            return "", CHAMPION

        # Route based on split ratio
        use_candidate = random.random() < ab_test.split_ratio
        variant = CANDIDATE if use_candidate else CHAMPION
        version = (
            ab_test.candidate_version if use_candidate
            else ab_test.control_version
        )

        pv = self._store.get_prompt_version(key, version)
        if pv is None:
            active = self._store.get_active_prompt(key)
            return (active.system_prompt if active else ""), CHAMPION

        return pv.system_prompt, variant

    # ── Recording results ────────────────────────────────────────────

    def record_result(
        self,
        variant_or_version: str | int,
        score: float,
        prompt_key: str = "",
        dimensions: dict[str, float] | None = None,
    ) -> None:
        """Record a feedback score for a variant in the active A/B test.

        Accepts either a variant label (``"champion"`` / ``"candidate"``)
        or a version number (for backward compatibility with code that
        tracks the integer version).

        Args:
            variant_or_version: ``"champion"``, ``"candidate"``, or an
                integer version number.
            score: Feedback score in [-1, 1].
            prompt_key: Override the default prompt namespace.
            dimensions: Per-dimension scores to track alongside composite.
        """
        key = prompt_key or self._prompt_key
        ab_test = self._store.get_active_ab_test(key)
        if ab_test is None:
            return

        # Resolve variant label → version number
        if isinstance(variant_or_version, str):
            if variant_or_version == CANDIDATE:
                version = ab_test.candidate_version
            elif variant_or_version == CHAMPION:
                version = ab_test.control_version
            else:
                return
        else:
            version = variant_or_version

        # Update running statistics
        if version == ab_test.control_version:
            n = ab_test.control_interactions + 1
            ab_test.control_score = (
                (ab_test.control_score * ab_test.control_interactions + score) / n
            )
            ab_test.control_interactions = n
            # Per-dimension tracking
            if dimensions:
                for dim_name, dim_score in dimensions.items():
                    old_count = ab_test.control_dimension_counts.get(dim_name, 0)
                    old_mean = ab_test.control_dimension_scores.get(dim_name, 0.0)
                    new_count = old_count + 1
                    ab_test.control_dimension_scores[dim_name] = (
                        (old_mean * old_count + dim_score) / new_count
                    )
                    ab_test.control_dimension_counts[dim_name] = new_count
        elif version == ab_test.candidate_version:
            n = ab_test.candidate_interactions + 1
            ab_test.candidate_score = (
                (ab_test.candidate_score * ab_test.candidate_interactions + score) / n
            )
            ab_test.candidate_interactions = n
            if dimensions:
                for dim_name, dim_score in dimensions.items():
                    old_count = ab_test.candidate_dimension_counts.get(dim_name, 0)
                    old_mean = ab_test.candidate_dimension_scores.get(dim_name, 0.0)
                    new_count = old_count + 1
                    ab_test.candidate_dimension_scores[dim_name] = (
                        (old_mean * old_count + dim_score) / new_count
                    )
                    ab_test.candidate_dimension_counts[dim_name] = new_count
        else:
            return

        self._store.update_ab_test(ab_test)

        # Auto-check after every update
        winner = self._check_and_resolve(ab_test)
        if winner:
            logger.info(
                "A/B test auto-resolved: %s wins for key=%s", winner, key,
            )

    # ── Statistical significance ─────────────────────────────────────

    def check_significance(self, prompt_key: str = "") -> str | None:
        """Check whether the A/B test has a statistically significant winner.

        Uses a two-sample Welch's t-test (implemented without scipy) at
        the configured significance level (default α = 0.05).

        Args:
            prompt_key: Override the default prompt namespace.

        Returns:
            - ``"candidate"`` if the candidate is significantly better.
            - ``"champion"`` if the champion is significantly better.
            - ``None`` if not enough data or no significant difference.
        """
        key = prompt_key or self._prompt_key
        ab_test = self._store.get_active_ab_test(key)
        if ab_test is None:
            return None

        return self._evaluate_significance(ab_test)

    def _evaluate_significance(self, ab_test: ABTest) -> str | None:
        """Core significance evaluation logic.

        When dimensions are configured:
        - Candidate is promoted only if composite is significantly better
          AND no high-priority dimension has significantly regressed.
        - If composite is better but a high-priority dimension is worse,
          returns "mixed" (no auto-promote).
        """
        n_champ = ab_test.control_interactions
        n_cand = ab_test.candidate_interactions

        if n_champ < self._min_interactions or n_cand < self._min_interactions:
            return None

        mean_champ = ab_test.control_score
        mean_cand = ab_test.candidate_score

        var_champ = max(0.01, (1.0 - abs(mean_champ)) * (1.0 + abs(mean_champ)) * 0.25)
        var_cand = max(0.01, (1.0 - abs(mean_cand)) * (1.0 + abs(mean_cand)) * 0.25)

        p_value = welch_ttest_p(
            mean_cand, var_cand, n_cand,
            mean_champ, var_champ, n_champ,
        )

        if p_value >= self._alpha:
            return None

        # Composite winner
        if mean_cand <= mean_champ:
            return CHAMPION

        # Check per-dimension regression for high-priority dimensions
        if (self._dimension_schema and self._dimension_schema.dimensions
                and ab_test.candidate_dimension_scores
                and ab_test.control_dimension_scores):
            for dim_name, dim in self._dimension_schema.dimensions.items():
                if dim.refinement_priority != "high":
                    continue
                cand_n = ab_test.candidate_dimension_counts.get(dim_name, 0)
                ctrl_n = ab_test.control_dimension_counts.get(dim_name, 0)
                # Only check with sufficient per-dimension samples
                if cand_n < 30 or ctrl_n < 30:
                    continue
                cand_mean = ab_test.candidate_dimension_scores.get(dim_name, 0.0)
                ctrl_mean = ab_test.control_dimension_scores.get(dim_name, 0.0)
                # Significant regression on high-priority dimension?
                if cand_mean < ctrl_mean - 0.1:
                    var_c = max(0.01, (1.0 - abs(ctrl_mean)) * (1.0 + abs(ctrl_mean)) * 0.25)
                    var_d = max(0.01, (1.0 - abs(cand_mean)) * (1.0 + abs(cand_mean)) * 0.25)
                    dim_p = welch_ttest_p(cand_mean, var_d, cand_n, ctrl_mean, var_c, ctrl_n)
                    if dim_p < self._alpha:
                        logger.warning(
                            "A/B test: candidate better overall but high-priority "
                            "dimension %s regressed (%.3f -> %.3f, p=%.4f) — "
                            "flagging as mixed",
                            dim_name, ctrl_mean, cand_mean, dim_p,
                        )
                        return "mixed"

        return CANDIDATE

    def _check_and_resolve(self, ab_test: ABTest) -> str | None:
        """Check significance and auto-promote/reject if there's a winner."""
        winner = self._evaluate_significance(ab_test)
        if winner is None:
            return None

        if winner == "mixed":
            # Do NOT auto-promote — flag as mixed
            ab_test.result = "mixed"
            self._store.update_ab_test(ab_test)
            logger.warning(
                "A/B test flagged as 'mixed' — manual decision required"
            )
            return "mixed"
        elif winner == CANDIDATE:
            self._promote(ab_test)
        else:
            self._reject(ab_test)

        return winner

    # ── Promotion / rejection ────────────────────────────────────────

    def promote_candidate(self, prompt_key: str = "") -> bool:
        """Promote the candidate in the active test to champion.

        Makes the candidate version the active prompt and closes the test.

        Returns:
            ``True`` if a candidate was promoted, ``False`` if no active test.
        """
        key = prompt_key or self._prompt_key
        ab_test = self._store.get_active_ab_test(key)
        if ab_test is None:
            return False
        self._promote(ab_test)
        return True

    def reject_candidate(self, prompt_key: str = "") -> bool:
        """Reject the candidate and keep the champion.

        Closes the test without changing the active prompt.

        Returns:
            ``True`` if a candidate was rejected, ``False`` if no active test.
        """
        key = prompt_key or self._prompt_key
        ab_test = self._store.get_active_ab_test(key)
        if ab_test is None:
            return False
        self._reject(ab_test)
        return True

    # ── Manual overrides (for dashboard) ─────────────────────────────

    def force_promote(self, ab_test_id: str | None = None) -> bool:
        """Manually promote the candidate, bypassing statistical checks.

        Args:
            ab_test_id: If provided, only acts if the active test
                matches this ID (prevents stale dashboard actions).

        Returns:
            ``True`` if the candidate was promoted, ``False`` otherwise.
        """
        ab_test = self._store.get_active_ab_test(self._prompt_key)
        if ab_test is None:
            return False
        if ab_test_id and ab_test.id != ab_test_id:
            return False
        self._promote(ab_test)
        logger.info("Force-promoted candidate v%d for key=%s",
                     ab_test.candidate_version, self._prompt_key)
        return True

    def force_reject(self, ab_test_id: str | None = None) -> bool:
        """Manually reject the candidate, bypassing statistical checks.

        Args:
            ab_test_id: If provided, only acts if the active test
                matches this ID.

        Returns:
            ``True`` if the candidate was rejected, ``False`` otherwise.
        """
        ab_test = self._store.get_active_ab_test(self._prompt_key)
        if ab_test is None:
            return False
        if ab_test_id and ab_test.id != ab_test_id:
            return False
        self._reject(ab_test)
        logger.info("Force-rejected candidate v%d for key=%s",
                     ab_test.candidate_version, self._prompt_key)
        return True

    # ── Querying ─────────────────────────────────────────────────────

    def get_active_test(self, prompt_key: str = "") -> ABTest | None:
        """Return the active A/B test for the given prompt_key, or None."""
        key = prompt_key or self._prompt_key
        return self._store.get_active_ab_test(key)

    def get_test_summary(self, prompt_key: str = "") -> dict[str, Any] | None:
        """Return a human-readable summary of the active test, or None."""
        key = prompt_key or self._prompt_key
        ab_test = self._store.get_active_ab_test(key)
        if ab_test is None:
            return None

        winner = self._evaluate_significance(ab_test)

        return {
            "test_id": ab_test.id,
            "prompt_key": key,
            "champion_version": ab_test.control_version,
            "candidate_version": ab_test.candidate_version,
            "split_ratio": ab_test.split_ratio,
            "champion_interactions": ab_test.control_interactions,
            "candidate_interactions": ab_test.candidate_interactions,
            "champion_score": round(ab_test.control_score, 4),
            "candidate_score": round(ab_test.candidate_score, 4),
            "significant_winner": winner,
            "min_interactions_remaining": max(
                0, self._min_interactions - ab_test.control_interactions,
                self._min_interactions - ab_test.candidate_interactions,
            ),
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _promote(self, ab_test: ABTest) -> None:
        """Make the candidate the active prompt and close the test."""
        key = ab_test.prompt_key or self._prompt_key
        self._store.set_active_version(key, ab_test.candidate_version)
        ab_test.is_active = False
        ab_test.completed_at = _utc_now()
        ab_test.result = "promoted"
        self._store.update_ab_test(ab_test)
        logger.info(
            "A/B test resolved: promoted candidate v%d "
            "(score %.3f, n=%d) over champion v%d (score %.3f, n=%d)",
            ab_test.candidate_version, ab_test.candidate_score,
            ab_test.candidate_interactions,
            ab_test.control_version, ab_test.control_score,
            ab_test.control_interactions,
        )

    def _reject(self, ab_test: ABTest) -> None:
        """Keep the champion and close the test."""
        ab_test.is_active = False
        ab_test.completed_at = _utc_now()
        ab_test.result = "rejected"
        self._store.update_ab_test(ab_test)
        logger.info(
            "A/B test resolved: rejected candidate v%d "
            "(score %.3f, n=%d) in favor of champion v%d (score %.3f, n=%d)",
            ab_test.candidate_version, ab_test.candidate_score,
            ab_test.candidate_interactions,
            ab_test.control_version, ab_test.control_score,
            ab_test.control_interactions,
        )
