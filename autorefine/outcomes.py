"""Outcome-based feedback — ground-truth scoring for deferred results.

For applications where ground truth is available after a delay (predictions,
recommendations, code generation with tests), the developer reports what
actually happened and AutoRefine translates this into dimensional feedback.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from autorefine.dimensions import FeedbackDimensionSchema
from autorefine.models import Interaction

logger = logging.getLogger("autorefine.outcomes")

# Confidence extraction patterns (conservative — better to return None
# than extract a wrong number)
_PCT_RE = re.compile(r"(\d{1,3})\s*%")
_PROB_RE = re.compile(r"probability[:\s]+(\d*\.?\d+)", re.IGNORECASE)
_CONF_WORD_RE = re.compile(
    r"confidence[:\s]+(very\s+high|very\s+low|high|medium|low)",
    re.IGNORECASE,
)
_OUT_OF_RE = re.compile(r"(\d*\.?\d+)\s*out\s+of\s*(\d*\.?\d+)", re.IGNORECASE)

_CONF_WORD_MAP = {
    "very high": 0.95,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2,
    "very low": 0.05,
}


class OutcomeTranslator:
    """Converts raw outcome reports into dimensional feedback scores.

    Subclass this to implement domain-specific outcome scoring.

    Default behavior:
    - correct=True: accuracy=+1.0, all others get +0.3
    - correct=False: accuracy=-1.0, calibration scored inversely to
      stated confidence, others get 0.0
    """

    def __init__(
        self, dimension_schema: FeedbackDimensionSchema | None = None
    ) -> None:
        self._schema = dimension_schema

    def translate(
        self,
        outcome: dict[str, Any],
        dimension_overrides: dict[str, float] | None = None,
        interaction: Interaction | None = None,
    ) -> dict[str, float]:
        """Convert an outcome dict into dimension scores.

        Args:
            outcome: Dict with 'predicted', 'actual', 'correct' keys.
            dimension_overrides: Developer-supplied scores that override
                auto-translation.
            interaction: The original Interaction, if available, so the
                translator can inspect the model's response for confidence.

        Returns:
            Dict mapping dimension name -> score in [-1, 1].
        """
        correct = outcome.get("correct", False)
        scores: dict[str, float] = {}

        if self._schema is None or not self._schema.dimensions:
            # No dimensions configured — return a simple score
            return {"_composite": 1.0 if correct else -1.0}

        # Extract confidence from response text
        confidence: float | None = None
        if interaction:
            confidence = self._extract_confidence_from_response(
                interaction.response_text
            )

        for dim_name in self._schema.dimensions:
            if dim_name == "accuracy":
                scores[dim_name] = 1.0 if correct else -1.0
            elif dim_name == "calibration":
                if correct:
                    scores[dim_name] = 0.3
                elif confidence is not None:
                    # Higher confidence when wrong = worse calibration
                    scores[dim_name] = -(confidence * 0.9)
                else:
                    scores[dim_name] = 0.0
                    logger.debug(
                        "No confidence found in response — calibration set to 0.0"
                    )
            else:
                scores[dim_name] = 0.3 if correct else 0.0

        # Apply overrides (developer knows best)
        if dimension_overrides:
            scores.update(dimension_overrides)

        return scores

    def _extract_confidence_from_response(
        self, response_text: str
    ) -> float | None:
        """Parse confidence level from the model's response text.

        Looks for patterns like '85% confident', 'probability: 0.7', etc.
        Returns None if no confidence found.
        """
        if not response_text:
            return None

        # Try percentage pattern: "85% chance", "85% confident"
        pct_matches = _PCT_RE.findall(response_text)
        if pct_matches:
            # Take the last percentage (most likely the confidence)
            val = int(pct_matches[-1])
            if 0 <= val <= 100:
                return val / 100.0

        # Try probability pattern: "probability: 0.7"
        prob_match = _PROB_RE.search(response_text)
        if prob_match:
            val = float(prob_match.group(1))
            if 0.0 <= val <= 1.0:
                return val

        # Try word pattern: "confidence: high"
        word_match = _CONF_WORD_RE.search(response_text)
        if word_match:
            word = word_match.group(1).lower().strip()
            return _CONF_WORD_MAP.get(word)

        # Try "X out of Y" pattern
        oof_match = _OUT_OF_RE.search(response_text)
        if oof_match:
            num = float(oof_match.group(1))
            denom = float(oof_match.group(2))
            if denom > 0:
                return max(0.0, min(1.0, num / denom))

        return None
