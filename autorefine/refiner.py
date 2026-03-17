"""The brain of the AutoRefine SDK — analyses feedback and produces improved prompts.

The :class:`Refiner` takes a batch of :class:`~autorefine.feedback.FeedbackBundle`
objects (interaction + feedback pairs), constructs a detailed meta-prompt,
sends it to a refiner LLM (Claude by default), parses the structured JSON
response, optionally validates the candidate by replaying past interactions,
and returns a :class:`~autorefine.models.RefinementResult`.

The meta-prompt is the most critical piece of the entire system.  It
instructs the refiner to be **surgical and conservative** — patching gaps
rather than rewriting from scratch — and to use conditional logic rather
than absolute rules so it doesn't break behaviors that are already working.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from autorefine._retry import retry_provider_call
from autorefine.exceptions import CostLimitExceeded, NoFeedbackError, RefinementError
from autorefine.feedback import FeedbackBundle
from autorefine.feedback_filter import FeedbackFilter
from autorefine.models import (
    CostEntry,
    FeedbackSignal,
    MessageRole,
    PromptCandidate,
    PromptVersion,
    RefinementResult,
)
from autorefine.pii_scrubber import PIIScrubber
from autorefine.providers.base import BaseProvider
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.refiner")


# ═══════════════════════════════════════════════════════════════════════
# Meta-prompt — the most important string in the entire SDK
# ═══════════════════════════════════════════════════════════════════════

META_PROMPT = """\
You are an expert prompt engineer performing a surgical revision of a system prompt.
Real users have interacted with an AI assistant that uses this prompt, and they have
provided feedback. Your job is to analyse their feedback, find the specific gaps in
the current prompt, and patch them — while preserving everything that already works.

═══════════════════════════════════════════════
CURRENT SYSTEM PROMPT (v{prompt_version})
═══════════════════════════════════════════════
{current_prompt}

{directives_block}\
{dimension_definitions}\
═══════════════════════════════════════════════
FEEDBACK OVERVIEW
═══════════════════════════════════════════════
{feedback_summary}

{dimension_analysis}\
═══════════════════════════════════════════════
INTERACTION EVIDENCE
═══════════════════════════════════════════════
Below are real interactions showing what users asked, how the model responded, and
how users reacted. Study them carefully — the negative feedback shows you EXACTLY
what the prompt is missing, and the positive feedback shows you what to PRESERVE.

{interaction_log}

═══════════════════════════════════════════════
YOUR INSTRUCTIONS
═══════════════════════════════════════════════

### Step 1: Analyse{step1_dimension_hint}
- Read every negative feedback signal. Group them into categories.
- For each category, identify the ROOT CAUSE in the current prompt.
  Ask: "What instruction is missing, ambiguous, or wrong that caused this failure?"
- Also note what the positive feedback tells you: which behaviors MUST be preserved.

### Step 2: Identify specific gaps
- List each gap as a concrete, testable statement.
  GOOD: "The prompt does not tell the model to ask clarifying questions when
         the user's request is ambiguous."
  BAD:  "The prompt needs to be better."

### Step 3: Rewrite surgically
You MUST follow these rules when rewriting:

1. **Patch, don't rewrite.** Start from the current prompt and make targeted edits.
   Do NOT discard the existing structure or rephrase things that are already working.

2. **Use conditional logic, not absolutes.** If feedback is mixed, add a conditional
   rule. NEVER swing to an extreme based on a subset of complaints.

3. **Preserve positive behaviors.** Before removing or weakening ANY instruction,
   verify that no positive feedback relied on it.

4. **Be specific and concrete.** Every instruction should be actionable.

5. **Respect the feedback distribution.** Weight your edits proportionally to the evidence.

6. **No meta-commentary.** The new prompt should contain ONLY instructions for the
   AI assistant — no notes about what you changed or why. Those go in the changelog.

### Step 4: Output
Respond with ONLY a JSON object. No markdown fences, no commentary outside the JSON.

{{
  "new_prompt": "<the complete rewritten system prompt>",
  "changelog": ["<specific change 1: what and why>", ...],
  "gaps_identified": ["<gap 1: concrete description>", ...],
  "reasoning": "<2-3 paragraphs explaining patterns, choices, trade-offs>",
  "expected_improvements": ["<improvement 1>", ...],
  "dimension_improvements": {{"<dimension_name>": "<what should improve>", ...}},
  "directives_respected": ["<directive text that was respected>", ...],
  "behaviors_preserved": ["<behavior that was preserved>", ...],
  "conflicts_detected": ["<tension between directive and feedback>", ...]
}}
"""

REFINER_SYSTEM_PROMPT = (
    "You are a world-class prompt engineer. You analyse user feedback on AI "
    "assistants and surgically improve their system prompts. You always respond "
    "with valid JSON — no markdown fences, no text outside the JSON object. "
    "You are conservative: you patch gaps rather than rewriting from scratch."
)


# ═══════════════════════════════════════════════════════════════════════
# Prompt-building helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_feedback_summary(feedback_items: list[FeedbackSignal]) -> str:
    """Produce a statistical summary so the refiner can gauge signal strength."""
    if not feedback_items:
        return "(no feedback)"

    total = len(feedback_items)
    positives = sum(1 for fb in feedback_items if fb.score > 0)
    negatives = sum(1 for fb in feedback_items if fb.score < 0)
    neutrals = total - positives - negatives
    avg_score = sum(fb.score for fb in feedback_items) / total
    avg_confidence = sum(fb.confidence for fb in feedback_items) / total

    corrections = sum(
        1 for fb in feedback_items if fb.feedback_type.value == "correction"
    )
    outcomes = sum(
        1 for fb in feedback_items if fb.feedback_type.value == "outcome"
    )

    lines = [
        f"Total signals: {total}",
        f"  Positive:    {positives:>3}  ({100 * positives / total:.0f}%)",
        f"  Negative:    {negatives:>3}  ({100 * negatives / total:.0f}%)",
    ]
    if neutrals:
        lines.append(
            f"  Neutral:     {neutrals:>3}  ({100 * neutrals / total:.0f}%)"
        )
    if corrections:
        lines.append(
            f"  Corrections: {corrections:>3}  (user-provided better answers)"
        )
    lines.append(f"Average score: {avg_score:+.2f}  |  Average confidence: {avg_confidence:.2f}")

    # Outcome data summary
    if outcomes:
        outcome_items = [fb for fb in feedback_items if fb.feedback_type.value == "outcome"]
        correct = sum(1 for fb in outcome_items if fb.context.get("correct", False))
        incorrect = outcomes - correct
        lines.append(f"\nOUTCOME DATA: {outcomes} of {total} signals include ground-truth outcomes.")
        lines.append(f"  Correct predictions:   {correct}/{outcomes} ({100 * correct / outcomes:.0f}%)")
        lines.append(f"  Incorrect predictions: {incorrect}/{outcomes} ({100 * incorrect / outcomes:.0f}%)")

    # Interpretation hint
    if positives > 0 and negatives / max(positives, 1) < 0.3:
        lines.append(
            "\n-> Mostly positive. Make small, targeted fixes only."
        )
    elif negatives > positives:
        lines.append(
            "\n-> More negative than positive. Consider structural improvements."
        )

    return "\n".join(lines)


def _build_interaction_log(
    bundles: list[FeedbackBundle],
    scrubber: PIIScrubber | None = None,
) -> str:
    """Format interaction/feedback bundles for the meta-prompt."""
    if not bundles:
        return "(no interactions with feedback)"

    entries: list[str] = []
    for i, bundle in enumerate(bundles, 1):
        ix = bundle.interaction
        user_msgs = [m.content for m in ix.messages if m.role == MessageRole.USER]
        user_text = user_msgs[-1] if user_msgs else "(no user message)"
        response_text = ix.response_text[:600]

        if scrubber:
            user_text = scrubber.scrub(user_text)
            response_text = scrubber.scrub(response_text)

        fb_lines: list[str] = []
        for fb in bundle.feedback:
            label = fb.feedback_type.value.upper()
            parts = [
                f"  [{label}] score={fb.score:+.1f} confidence={fb.confidence:.1f}"
            ]

            # Dimensional scores
            if fb.dimensions:
                dim_str = "  ".join(
                    f"{k}: {v:+.1f}" for k, v in fb.dimensions.items()
                )
                parts.append(f"  DIMENSION SCORES: {dim_str}")

            if fb.comment:
                comment = scrubber.scrub(fb.comment) if scrubber else fb.comment
                parts.append(f'    User said: "{comment}"')
            if fb.correction:
                correction = scrubber.scrub(fb.correction) if scrubber else fb.correction
                parts.append(f'    User\'s preferred answer: "{correction}"')

            # Developer context
            ctx = fb.context
            if ctx:
                if scrubber:
                    ctx = {k: scrubber.scrub(str(v)) for k, v in ctx.items()}
                ctx_lines = [f"    {k}: {v}" for k, v in ctx.items()]
                parts.append("  DEVELOPER CONTEXT:\n" + "\n".join(ctx_lines))

            # Outcome data
            if fb.feedback_type.value == "outcome" and ctx:
                predicted = ctx.get("predicted", "")
                actual = ctx.get("actual", "")
                correct = ctx.get("correct", "")
                if predicted or actual:
                    outcome_label = "CORRECT" if correct else "INCORRECT"
                    parts.append(
                        f"  OUTCOME: {outcome_label} "
                        f"(predicted: {predicted}, actual: {actual})"
                    )

            fb_lines.append("\n".join(parts))

        entry = (
            f"--- Interaction #{i} (id: {ix.id[:8]}, prompt v{ix.prompt_version}) ---\n"
            f"USER: {user_text}\n"
            f"MODEL: {response_text}\n"
            f"FEEDBACK:\n" + ("\n".join(fb_lines) if fb_lines else "  (none)")
        )
        entries.append(entry)

    return "\n\n".join(entries)


# ═══════════════════════════════════════════════════════════════════════
# The Refiner
# ═══════════════════════════════════════════════════════════════════════

class Refiner:
    """Analyses feedback bundles and produces improved prompt candidates.

    The refiner is the brain of the SDK.  It:

    1. Gathers unprocessed feedback (or accepts pre-built bundles)
    2. Constructs a detailed meta-prompt with the current prompt + evidence
    3. Sends the meta-prompt to a refiner LLM (Claude by default)
    4. Parses the structured JSON response
    5. Optionally validates by replaying past interactions
    6. Returns a :class:`RefinementResult` and/or :class:`PromptCandidate`

    Args:
        refiner_provider: The LLM provider for refinement calls.
        store: Storage backend for reading interactions/feedback.
        prompt_key: Default prompt namespace.
        batch_size: Max feedback signals per refinement cycle.
        cost_limit: Monthly USD cap for refiner calls.
        pii_scrubber: Optional PII redactor.
        feedback_filter: Optional noise filter.
        validation_count: Number of past interactions to replay for
            validation (0 to disable).
    """

    def __init__(
        self,
        refiner_provider: BaseProvider,
        store: BaseStore,
        prompt_key: str = "default",
        batch_size: int = 50,
        cost_limit: float = 25.0,
        pii_scrubber: PIIScrubber | None = None,
        feedback_filter: FeedbackFilter | None = None,
        validation_count: int = 3,
        dimension_schema: Any = None,
        directive_manager: Any = None,
    ) -> None:
        self._provider = refiner_provider
        self._store = store
        self._prompt_key = prompt_key
        self._batch_size = batch_size
        self._cost_limit = cost_limit
        self._scrubber = pii_scrubber
        self._feedback_filter = feedback_filter
        self._validation_count = validation_count
        self._dimension_schema = dimension_schema
        self._directive_manager = directive_manager

    # ── Primary API ──────────────────────────────────────────────────

    def refine(
        self,
        prompt_key: str = "",
        current_prompt: str = "",
        feedback_bundles: list[FeedbackBundle] | None = None,
    ) -> PromptCandidate:
        """Run a full refinement cycle and return a candidate prompt.

        Can be called in two ways:

        1. **No-args** (used by ``client._run_refinement``): gathers
           feedback from the store automatically.
        2. **With arguments**: uses the provided bundles directly.

        Args:
            prompt_key: Prompt namespace.  Falls back to the default.
            current_prompt: The current system prompt text.  If empty,
                looked up from the store.
            feedback_bundles: Pre-built bundles.  If ``None``, gathered
                from the store.

        Returns:
            A :class:`PromptCandidate` ready for A/B testing or promotion.

        Raises:
            CostLimitExceeded: Monthly refiner budget exhausted.
            NoFeedbackError: No actionable feedback available.
            RefinementError: Refiner returned invalid/unparseable output.
        """
        key = prompt_key or self._prompt_key

        # ── Cost guard ──
        self._check_cost_limit()

        # ── Resolve current prompt ──
        active = self._store.get_active_prompt(key)
        if not current_prompt:
            current_prompt = active.system_prompt if active else ""
        current_version = active.version if active else 0

        # ── Gather feedback bundles ──
        if feedback_bundles is None:
            feedback_bundles = self._gather_bundles(key)

        if not feedback_bundles:
            raise NoFeedbackError("No unprocessed feedback available for refinement")

        # ── Build and send meta-prompt ──
        all_feedback = [fb for b in feedback_bundles for fb in b.feedback]
        result = self._call_refiner(
            current_prompt, current_version, feedback_bundles, all_feedback
        )

        # ── Track cost was already done inside _call_refiner ──

        # ── Mark feedback as processed ──
        feedback_ids = [fb.id for b in feedback_bundles for fb in b.feedback]
        try:
            self._store.mark_feedback_processed(feedback_ids)
        except Exception:
            logger.warning("Failed to mark %d feedback IDs as processed", len(feedback_ids), exc_info=True)

        # ── Validate candidate (optional) ──
        if self._validation_count > 0:
            self._validate_candidate(
                result, current_prompt, feedback_bundles, key
            )

        # ── Build candidate ──
        directives_version = 0
        if self._directive_manager:
            rd = self._directive_manager.get(key)
            if rd:
                directives_version = rd.version

        candidate = PromptCandidate(
            prompt_key=key,
            system_prompt=result.new_prompt,
            parent_version=current_version,
            changelog=result.changelog if isinstance(result.changelog, str)
                else "; ".join(result.changelog) if isinstance(result.changelog, list)
                else str(result.changelog),
            reasoning=result.reasoning,
            expected_improvements=result.expected_improvements,
            directives_version=directives_version,
        )

        logger.info(
            "Refinement complete for key=%s (parent v%d, %d gaps found)",
            key, current_version, len(result.gaps_identified),
        )
        return candidate

    def promote_candidate(self, candidate: PromptCandidate) -> PromptVersion:
        """Promote a candidate to the next active prompt version."""
        key = candidate.prompt_key or self._prompt_key
        history = self._store.get_prompt_history(key)
        next_version = max((pv.version for pv in history), default=0) + 1

        version = PromptVersion(
            version=next_version,
            prompt_key=key,
            system_prompt=candidate.system_prompt,
            parent_version=candidate.parent_version,
            changelog=candidate.changelog,
            is_active=True,
        )

        self._store.set_active_version(key, -1)
        self._store.save_prompt_version(version)
        logger.info("Promoted prompt v%d for key=%s", next_version, key)
        return version

    # ── Internal: gathering ──────────────────────────────────────────

    def _gather_bundles(self, prompt_key: str) -> list[FeedbackBundle]:
        """Pull unprocessed feedback from the store and bundle with interactions."""
        feedback_items = self._store.get_feedback(
            prompt_key=prompt_key,
            unprocessed_only=True,
            limit=self._batch_size,
        )
        if not feedback_items:
            return []

        # Apply noise filter
        if self._feedback_filter:
            feedback_items = self._feedback_filter.filter(feedback_items)
            if not feedback_items:
                return []

        # Group by interaction
        fb_by_ix: dict[str, list[FeedbackSignal]] = {}
        for fb in feedback_items:
            fb_by_ix.setdefault(fb.interaction_id, []).append(fb)

        bundles: list[FeedbackBundle] = []
        for ix_id, fb_list in fb_by_ix.items():
            ix = self._store.get_interaction(ix_id)
            if ix:
                bundles.append(FeedbackBundle(interaction=ix, feedback=fb_list))

        return bundles

    # ── Internal: calling the refiner ────────────────────────────────

    def _call_refiner(
        self,
        current_prompt: str,
        current_version: int,
        bundles: list[FeedbackBundle],
        all_feedback: list[FeedbackSignal],
    ) -> RefinementResult:
        """Construct the meta-prompt, call the refiner, parse the response."""
        interaction_log = _build_interaction_log(bundles, scrubber=self._scrubber)
        feedback_summary = _build_feedback_summary(all_feedback)

        # ── Directives block ──
        directives_block = ""
        if self._directive_manager:
            directives_block = self._directive_manager.format_for_meta_prompt(
                self._prompt_key
            )
            if directives_block:
                directives_block += "\n"

        # ── Dimension definitions + analysis ──
        dimension_definitions = ""
        dimension_analysis = ""
        step1_hint = " failure patterns"
        if self._dimension_schema and self._dimension_schema.dimensions:
            from autorefine.dimensions import DimensionAggregator

            # Definitions table
            dim_lines = [
                "═══════════════════════════════════════════════",
                "FEEDBACK DIMENSION DEFINITIONS",
                "═══════════════════════════════════════════════",
                "",
            ]
            for name, dim in self._dimension_schema.dimensions.items():
                dim_lines.append(
                    f"  {name:<20} weight={dim.weight}  priority={dim.refinement_priority}"
                )
                dim_lines.append(f"    {dim.description}")
                dim_lines.append("")
            dimension_definitions = "\n".join(dim_lines) + "\n"

            # Per-dimension analysis
            aggregator = DimensionAggregator(self._dimension_schema)
            dimension_analysis = aggregator.format_for_meta_prompt(all_feedback)
            if dimension_analysis:
                dimension_analysis = (
                    "═══════════════════════════════════════════════\n"
                    "DIMENSION-LEVEL ANALYSIS\n"
                    "═══════════════════════════════════════════════\n"
                    + dimension_analysis + "\n\n"
                )
            step1_hint = " by dimension (not just overall sentiment)"

        meta_prompt = META_PROMPT.format(
            current_prompt=current_prompt or "(no system prompt set)",
            prompt_version=current_version,
            interaction_log=interaction_log,
            feedback_summary=feedback_summary,
            directives_block=directives_block,
            dimension_definitions=dimension_definitions,
            dimension_analysis=dimension_analysis,
            step1_dimension_hint=step1_hint,
        )

        logger.info(
            "Calling refiner model for prompt_key=%s (%d bundles, %d signals)",
            self._prompt_key, len(bundles), len(all_feedback),
        )

        resp = retry_provider_call(
            self._provider.complete, REFINER_SYSTEM_PROMPT, meta_prompt,
        )

        # Track cost
        cost = self._provider.estimate_cost(resp.input_tokens, resp.output_tokens)
        try:
            self._store.save_cost_entry(
                CostEntry(
                    model=resp.model,
                    provider=self._provider.name,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    cost_usd=cost,
                    call_type="refiner",
                )
            )
        except Exception:
            logger.warning("Failed to save refiner cost entry", exc_info=True)

        return self._parse_refiner_response(resp.text, feedback_summary)

    # ── Internal: validation ─────────────────────────────────────────

    def _validate_candidate(
        self,
        result: RefinementResult,
        original_prompt: str,
        bundles: list[FeedbackBundle],
        prompt_key: str,
    ) -> None:
        """Replay a few past interactions through the new prompt and log results.

        This is a lightweight sanity check: we re-send user messages through
        the refiner provider using the NEW prompt and compare.  We don't
        block promotion on the result — we just log warnings if the new
        prompt produces suspiciously short or empty responses.

        The cost of these validation calls is tracked as refiner spend.
        """
        # Pick interactions from the bundles that had negative feedback
        negative_bundles = [
            b for b in bundles
            if any(fb.score < 0 for fb in b.feedback)
        ]
        candidates = negative_bundles[:self._validation_count]

        if not candidates:
            return

        logger.info(
            "Validating candidate prompt against %d past interactions",
            len(candidates),
        )

        for bundle in candidates:
            user_msgs = [
                m for m in bundle.interaction.messages
                if m.role == MessageRole.USER
            ]
            if not user_msgs:
                continue

            try:
                resp = self._provider.chat(
                    result.new_prompt,
                    user_msgs,
                )

                # Track validation cost
                cost = self._provider.estimate_cost(
                    resp.input_tokens, resp.output_tokens
                )
                self._store.save_cost_entry(
                    CostEntry(
                        model=resp.model,
                        provider=self._provider.name,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        cost_usd=cost,
                        call_type="refiner",
                    )
                )

                # Sanity checks
                if not resp.text or not resp.text.strip():
                    logger.warning(
                        "Validation: new prompt produced EMPTY response for "
                        "interaction %s — candidate may be flawed",
                        bundle.interaction.id[:8],
                    )
                elif len(resp.text) < len(bundle.interaction.response_text) * 0.1:
                    logger.warning(
                        "Validation: new prompt produced a response 90%%+ shorter "
                        "than the original for interaction %s (%d vs %d chars)",
                        bundle.interaction.id[:8],
                        len(resp.text),
                        len(bundle.interaction.response_text),
                    )
                else:
                    logger.debug(
                        "Validation passed for interaction %s (%d chars)",
                        bundle.interaction.id[:8], len(resp.text),
                    )

            except Exception:
                logger.warning(
                    "Validation call failed for interaction %s — "
                    "continuing without validation",
                    bundle.interaction.id[:8],
                    exc_info=True,
                )

    # ── Internal: cost guard ─────────────────────────────────────────

    def _check_cost_limit(self) -> None:
        """Raise if the monthly refiner budget is exhausted."""
        monthly_cost = self._store.get_monthly_refiner_cost()
        if monthly_cost >= self._cost_limit:
            raise CostLimitExceeded(
                f"Monthly refiner cost ${monthly_cost:.2f} exceeds "
                f"limit ${self._cost_limit:.2f}",
                current_spend=monthly_cost,
                limit=self._cost_limit,
            )

    # ── Internal: response parsing ───────────────────────────────────

    @staticmethod
    def _parse_refiner_response(
        text: str,
        feedback_summary: str = "",
    ) -> RefinementResult:
        """Parse the refiner model's JSON response with robust error handling.

        Handles:
        - Markdown code fences (````` ```json ... ``` `````)
        - Leading/trailing whitespace
        - Missing optional fields (graceful defaults)
        - Completely malformed JSON (clear error message)
        - Missing required ``new_prompt`` field
        """
        cleaned = text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove opening fence (```json or ```)
            if lines:
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Try to find JSON object if there's surrounding text
        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Provide a helpful error with context
            preview = text[:200].replace("\n", "\\n")
            raise RefinementError(
                f"Refiner returned invalid JSON: {exc}. "
                f"Response preview: {preview!r}",
            ) from exc

        if not isinstance(data, dict):
            raise RefinementError(
                f"Refiner returned {type(data).__name__} instead of a JSON object"
            )

        if "new_prompt" not in data:
            keys = list(data.keys())[:10]
            raise RefinementError(
                f"Refiner response missing required 'new_prompt' field. "
                f"Keys found: {keys}"
            )

        new_prompt = data["new_prompt"]
        if not isinstance(new_prompt, str) or not new_prompt.strip():
            raise RefinementError(
                "Refiner returned an empty or non-string 'new_prompt'"
            )

        # Normalize changelog: accept string or list
        raw_changelog = data.get("changelog", "")
        if isinstance(raw_changelog, list):
            changelog = "; ".join(str(c) for c in raw_changelog)
        else:
            changelog = str(raw_changelog)

        # Normalize gaps_identified: accept string or list
        raw_gaps = data.get("gaps_identified", [])
        if isinstance(raw_gaps, str):
            gaps = [raw_gaps] if raw_gaps else []
        elif isinstance(raw_gaps, list):
            gaps = [str(g) for g in raw_gaps]
        else:
            gaps = []

        # Normalize expected_improvements
        raw_improvements = data.get("expected_improvements", [])
        if isinstance(raw_improvements, str):
            improvements = [raw_improvements] if raw_improvements else []
        elif isinstance(raw_improvements, list):
            improvements = [str(i) for i in raw_improvements]
        else:
            improvements = []

        # Normalize dimension_improvements
        raw_dim_imp = data.get("dimension_improvements", {})
        dim_improvements = {}
        if isinstance(raw_dim_imp, dict):
            dim_improvements = {str(k): str(v) for k, v in raw_dim_imp.items()}

        # Normalize list fields
        directives_respected = data.get("directives_respected", [])
        if isinstance(directives_respected, list):
            directives_respected = [str(d) for d in directives_respected]
        else:
            directives_respected = []

        behaviors_preserved = data.get("behaviors_preserved", [])
        if isinstance(behaviors_preserved, list):
            behaviors_preserved = [str(b) for b in behaviors_preserved]
        else:
            behaviors_preserved = []

        conflicts_detected = data.get("conflicts_detected", [])
        if isinstance(conflicts_detected, list):
            conflicts_detected = [str(c) for c in conflicts_detected]
        else:
            conflicts_detected = []

        return RefinementResult(
            new_prompt=new_prompt.strip(),
            changelog=changelog,
            reasoning=str(data.get("reasoning", "")),
            gaps_identified=gaps,
            expected_improvements=improvements,
            feedback_summary=feedback_summary,
            dimension_improvements=dim_improvements,
            directives_respected=directives_respected,
            behaviors_preserved=behaviors_preserved,
            conflicts_detected=conflicts_detected,
        )
