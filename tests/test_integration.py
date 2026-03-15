"""Full end-to-end integration test — proves the entire learning loop works
without any real API calls.

Flow: client.chat → feedback → threshold → refine → A/B test → promote → new prompt active.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from autorefine import AutoRefine
from autorefine.ab_testing import CANDIDATE, CHAMPION
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.storage.json_store import JSONStore

# ═══════════════════════════════════════════════════════════════════════
# Mock providers
# ═══════════════════════════════════════════════════════════════════════

class MockLLMProvider(BaseProvider):
    """Canned LLM that echoes the system prompt so tests can verify which
    prompt version is in effect."""

    name = "mock-llm"

    def __init__(self):
        self.call_count = 0
        self.last_system_prompt = ""

    def chat(self, system_prompt, messages, **kwargs):
        self.call_count += 1
        self.last_system_prompt = system_prompt
        return ProviderResponse(
            text=f"Response #{self.call_count} (prompt: {system_prompt[:40]}...)",
            input_tokens=50,
            output_tokens=30,
            model="mock-llm-v1",
        )

    def stream(self, system_prompt, messages, **kwargs):
        yield f"streamed: {system_prompt[:20]}"

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001


class MockRefinerProvider(BaseProvider):
    """Deterministic refiner that analyzes feedback comments and produces
    a hardcoded improvement.

    If feedback contains "too verbose" or "too long", adds "Be concise."
    If feedback contains "inaccurate", adds "Cite sources."
    """

    name = "mock-refiner"

    def __init__(self):
        self.call_count = 0

    def chat(self, system_prompt, messages, **kwargs):
        self.call_count += 1
        user_text = messages[-1].content if messages else ""

        # Parse the current prompt from the meta-prompt
        current_prompt = "You are a helpful assistant."
        if "CURRENT SYSTEM PROMPT" in user_text:
            # Extract between the markers
            start = user_text.find("CURRENT SYSTEM PROMPT")
            end = user_text.find("FEEDBACK OVERVIEW")
            if start != -1 and end != -1:
                block = user_text[start:end]
                # The prompt is in the block between the header lines
                lines = block.split("\n")
                prompt_lines = [
                    line for line in lines
                    if line.strip() and "CURRENT SYSTEM PROMPT" not in line
                    and "═" not in line
                ]
                if prompt_lines:
                    current_prompt = "\n".join(prompt_lines).strip()

        # Analyze feedback to decide what to add
        additions = []
        if "verbose" in user_text.lower() or "too long" in user_text.lower():
            additions.append("Be concise and direct.")
        if "inaccurate" in user_text.lower():
            additions.append("Cite sources when possible.")

        new_prompt = current_prompt
        for addition in additions:
            if addition not in new_prompt:
                new_prompt = new_prompt.rstrip(".") + ". " + addition

        response = json.dumps({
            "new_prompt": new_prompt,
            "changelog": ["Added conciseness instructions"] if additions else ["No changes needed"],
            "gaps_identified": ["Missing conciseness directive"] if additions else [],
            "reasoning": "Users complained about verbosity, so I added conciseness instructions.",
            "expected_improvements": ["Shorter, more direct responses"],
        })

        return ProviderResponse(
            text=response,
            input_tokens=500,
            output_tokens=300,
            model="mock-refiner-v1",
        )

    def stream(self, system_prompt, messages, **kwargs):
        yield '{"new_prompt": "test"}'

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.01


# ═══════════════════════════════════════════════════════════════════════
# Integration test
# ═══════════════════════════════════════════════════════════════════════

class TestFullLearningLoop:
    """Proves the complete cycle: chat → feedback → refine → A/B → promote."""

    @pytest.fixture
    def env(self):
        """Set up a fully wired AutoRefine client with mock providers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONStore(str(Path(tmpdir) / "integration.json"))
            llm = MockLLMProvider()
            refiner = MockRefinerProvider()

            with patch("autorefine.client.get_provider") as mock_get:
                # First call = primary provider, second = refiner provider
                mock_get.side_effect = [llm, refiner]
                client = AutoRefine(
                    api_key="test-key",
                    model="mock",
                    refiner_key="refiner-key",
                    auto_learn=False,  # we'll trigger manually
                    store=store,
                    refine_threshold=20,
                    ab_test_split=0.5,
                    ab_test_min_interactions=10,
                    feedback_filter_enabled=False,
                    pii_scrub_enabled=False,
                )

            # Replace providers on the interceptor too
            client._provider = llm
            client._interceptor._provider = llm
            client._refiner._provider = refiner
            client._refiner._validation_count = 0  # skip validation calls

            yield client, store, llm, refiner

    def test_step1_chat_and_collect_feedback(self, env):
        """Make 25 calls, submit negative feedback on 20."""
        client, store, llm, _ = env

        client.set_system_prompt("You are a helpful assistant.")

        # Make 25 chat calls
        response_ids = []
        for i in range(25):
            resp = client.chat("", [{"role": "user", "content": f"Question {i}"}])
            response_ids.append(resp.id)
            assert resp.text is not None
            assert resp.id

        assert llm.call_count == 25

        # Submit negative feedback on 20 of them
        for i in range(20):
            comment = "too verbose" if i % 2 == 0 else "response is too long"
            client.feedback(response_ids[i], "thumbs_down", comment=comment)

        # Submit positive feedback on the remaining 5
        for i in range(20, 25):
            client.feedback(response_ids[i], "thumbs_up")

        # Verify feedback stored
        all_fb = store.get_feedback(prompt_key="default", limit=100)
        assert len(all_fb) == 25

    def test_step2_threshold_detection(self, env):
        """Verify should_trigger_refinement returns True after enough feedback."""
        client, store, _, _ = env
        client.set_system_prompt("You are a helpful assistant.")

        # Make calls and submit feedback
        for i in range(20):
            resp = client.chat("", [{"role": "user", "content": f"Q{i}"}])
            client.feedback(resp.id, "thumbs_down", comment="too verbose")

        assert client._feedback.should_trigger_refinement() is True

    def test_step3_refinement_produces_concise_prompt(self, env):
        """Trigger refinement and verify the candidate includes conciseness."""
        client, store, _, refiner = env
        client.set_system_prompt("You are a helpful assistant.")

        # Accumulate feedback
        for i in range(20):
            resp = client.chat("", [{"role": "user", "content": f"Q{i}"}])
            client.feedback(resp.id, "thumbs_down", comment="too verbose and rambling")

        # Trigger refinement (returns None because A/B test starts)
        result = client.refine_now()

        # With ab_test_split=0.5, refinement starts an A/B test
        assert result is None  # candidate entered A/B testing

        # Verify the A/B test was created
        ab_test = store.get_active_ab_test("default")
        assert ab_test is not None
        assert ab_test.is_active is True

        # Verify the candidate prompt contains conciseness
        candidate_pv = store.get_prompt_version("default", ab_test.candidate_version)
        assert candidate_pv is not None
        assert "concise" in candidate_pv.system_prompt.lower() or \
               "direct" in candidate_pv.system_prompt.lower()

        # Verify refiner was called
        assert refiner.call_count >= 1

    def test_step4_full_loop_through_promotion(self, env):
        """Complete end-to-end: chat → feedback → refine → A/B → promote."""
        client, store, llm, refiner = env
        client.set_system_prompt("You are a helpful assistant.")

        # ── Phase 1: Accumulate negative feedback ──
        for i in range(20):
            resp = client.chat("", [{"role": "user", "content": f"Q{i}"}])
            client.feedback(resp.id, "thumbs_down", comment="too verbose")

        # ── Phase 2: Trigger refinement → enters A/B test ──
        client.refine_now()
        ab_test = store.get_active_ab_test("default")
        assert ab_test is not None
        candidate_v = ab_test.candidate_version

        # ── Phase 3: Simulate A/B test with candidate winning ──
        # Record enough results for both variants to reach min_interactions
        for _ in range(15):
            # Champion gets mediocre scores
            client._ab.record_result(CHAMPION, -0.3)
            # Candidate gets great scores
            client._ab.record_result(CANDIDATE, 0.9)

        # ── Phase 4: Verify candidate was promoted ──
        active = store.get_active_prompt("default")
        assert active is not None
        assert active.version == candidate_v
        assert "concise" in active.system_prompt.lower() or \
               "direct" in active.system_prompt.lower()

        # ── Phase 5: Verify next call uses the new prompt ──
        resp = client.chat("", [{"role": "user", "content": "Final question"}])
        assert resp.text is not None
        # The interceptor should have picked up the new active prompt
        assert llm.last_system_prompt != "You are a helpful assistant."

        # ── Phase 6: Verify history ──
        history = client.get_prompt_history()
        assert len(history) >= 2

        # Verify feedback was marked processed
        unprocessed = store.get_feedback(prompt_key="default", unprocessed_only=True)
        # Some may remain from the A/B phase
        assert len(unprocessed) < 20  # most were consumed by refinement

    def test_auto_learn_triggers_refinement(self, env):
        """Test that auto_learn=True triggers refinement at threshold."""
        client, store, _, _ = env

        # Re-enable auto_learn by wiring the callback
        client._feedback._on_ready = client._run_refinement
        client._cfg.auto_learn = True
        client.set_system_prompt("You are a helpful assistant.")

        # Submit feedback up to threshold (20)
        for i in range(20):
            resp = client.chat("", [{"role": "user", "content": f"Q{i}"}])
            client.feedback(resp.id, "thumbs_down", comment="too verbose")

        # Auto-learn should have triggered refinement → A/B test started
        ab_test = store.get_active_ab_test("default")
        assert ab_test is not None


class TestRefinementWithoutABTest:
    """Test direct promotion when ab_test_split=0."""

    def test_direct_promotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONStore(str(Path(tmpdir) / "direct.json"))
            llm = MockLLMProvider()
            refiner = MockRefinerProvider()

            with patch("autorefine.client.get_provider") as mock_get:
                mock_get.side_effect = [llm, refiner]
                client = AutoRefine(
                    api_key="test", model="mock", refiner_key="refiner-key",
                    store=store, refine_threshold=5,
                    ab_test_split=0.0,  # skip A/B testing
                    feedback_filter_enabled=False,
                    pii_scrub_enabled=False,
                )

            client._provider = llm
            client._interceptor._provider = llm
            client._refiner._provider = refiner
            client._refiner._validation_count = 0

            client.set_system_prompt("You are a helpful assistant.")

            # Accumulate feedback
            for i in range(5):
                resp = client.chat("", [{"role": "user", "content": f"Q{i}"}])
                client.feedback(resp.id, "thumbs_down", comment="too verbose")

            # Refine — should promote directly (no A/B)
            version = client.refine_now()
            assert version is not None
            assert version.version == 2
            assert "concise" in version.system_prompt.lower() or \
                   "direct" in version.system_prompt.lower()
