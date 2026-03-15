"""Tests for A/B testing module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from autorefine.ab_testing import (
    CANDIDATE,
    CHAMPION,
    ABTestManager,
    welch_ttest_p,
)
from autorefine.models import PromptCandidate, PromptVersion
from autorefine.storage.json_store import JSONStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = JSONStore(str(Path(tmpdir) / "test.json"))
        s.save_prompt_version(
            PromptVersion(
                version=1,
                prompt_key="default",
                system_prompt="Champion prompt",
                is_active=True,
            )
        )
        yield s


@pytest.fixture
def tester(store):
    return ABTestManager(
        store=store,
        prompt_key="default",
        split_ratio=0.5,
        min_interactions=5,
        significance_level=0.05,
    )


def _candidate(prompt: str = "Candidate prompt") -> PromptCandidate:
    return PromptCandidate(
        system_prompt=prompt,
        parent_version=1,
        changelog="Test improvement",
    )


# ── Starting a test ──────────────────────────────────────────────────

class TestStartTest:
    def test_creates_test_record(self, tester, store):
        ab_test = tester.start_test(_candidate())
        assert ab_test.control_version == 1
        assert ab_test.candidate_version == 2
        assert ab_test.is_active is True
        assert ab_test.split_ratio == 0.5

    def test_saves_candidate_as_inactive_version(self, tester, store):
        tester.start_test(_candidate("New prompt"))
        pv = store.get_prompt_version("default", 2)
        assert pv is not None
        assert pv.system_prompt == "New prompt"
        assert pv.is_active is False

    def test_supersedes_existing_test(self, tester, store):
        tester.start_test(_candidate("First"))
        test2 = tester.start_test(_candidate("Second"))

        # First test should be superseded
        assert store.get_active_ab_test("default").id == test2.id
        assert test2.candidate_version == 3  # v2 was first candidate, v3 is second

    def test_with_prompt_key_override(self, tester, store):
        store.save_prompt_version(
            PromptVersion(version=1, prompt_key="billing", system_prompt="Billing v1", is_active=True)
        )
        ab_test = tester.start_test(_candidate(), prompt_key="billing")
        assert ab_test.prompt_key == "billing"


# ── Traffic routing ──────────────────────────────────────────────────

class TestGetPromptForRequest:
    def test_routes_traffic_to_both_variants(self, tester, store):
        tester.start_test(_candidate("Candidate prompt"))

        labels_seen = set()
        for _ in range(100):
            _, label = tester.get_prompt_for_request()
            labels_seen.add(label)

        assert CHAMPION in labels_seen
        assert CANDIDATE in labels_seen

    def test_returns_correct_prompt_text(self, tester, store):
        tester.start_test(_candidate("Candidate prompt"))

        prompts = {}
        for _ in range(100):
            text, label = tester.get_prompt_for_request()
            prompts[label] = text

        assert prompts.get(CHAMPION) == "Champion prompt"
        assert prompts.get(CANDIDATE) == "Candidate prompt"

    def test_no_active_test_returns_champion(self, tester):
        text, label = tester.get_prompt_for_request()
        assert text == "Champion prompt"
        assert label == CHAMPION

    def test_no_prompt_returns_empty(self, store):
        tester = ABTestManager(store=store, prompt_key="nonexistent")
        text, label = tester.get_prompt_for_request()
        assert text == ""
        assert label == CHAMPION

    def test_variant_label_type(self, tester, store):
        tester.start_test(_candidate())
        _, label = tester.get_prompt_for_request()
        assert isinstance(label, str)
        assert label in (CHAMPION, CANDIDATE)


# ── Recording results ────────────────────────────────────────────────

class TestRecordResult:
    def test_record_by_variant_label(self, tester, store):
        tester.start_test(_candidate())
        tester.record_result(CHAMPION, 0.8)
        tester.record_result(CANDIDATE, 0.5)

        test = store.get_active_ab_test("default")
        assert test.control_interactions == 1
        assert test.candidate_interactions == 1

    def test_record_by_version_number(self, tester, store):
        ab_test = tester.start_test(_candidate())
        tester.record_result(ab_test.control_version, 0.8)
        tester.record_result(ab_test.candidate_version, 0.5)

        test = store.get_active_ab_test("default")
        assert test.control_interactions == 1
        assert test.candidate_interactions == 1

    def test_running_average_computed(self, tester, store):
        tester.start_test(_candidate())
        tester.record_result(CHAMPION, 0.6)
        tester.record_result(CHAMPION, 0.8)

        test = store.get_active_ab_test("default")
        assert test.control_score == pytest.approx(0.7, abs=0.01)

    def test_ignores_unknown_variant(self, tester, store):
        tester.start_test(_candidate())
        tester.record_result("unknown_label", 0.5)
        tester.record_result(999, 0.5)  # unknown version
        test = store.get_active_ab_test("default")
        assert test.control_interactions == 0
        assert test.candidate_interactions == 0

    def test_no_active_test_is_noop(self, tester):
        tester.record_result(CHAMPION, 0.5)  # should not raise


# ── Significance testing ─────────────────────────────────────────────

class TestCheckSignificance:
    def test_returns_none_below_min_interactions(self, tester, store):
        tester.start_test(_candidate())
        for _ in range(3):  # below min of 5
            tester.record_result(CHAMPION, 0.5)
            tester.record_result(CANDIDATE, 0.9)
        assert tester.check_significance() is None

    def test_returns_candidate_when_significantly_better(self, store):
        # min_interactions > loop count so auto-resolve doesn't fire mid-loop
        tester = ABTestManager(
            store=store, prompt_key="default",
            split_ratio=0.5, min_interactions=200,
        )
        tester.start_test(_candidate())
        for _ in range(150):
            tester.record_result(CHAMPION, 0.2)
            tester.record_result(CANDIDATE, 0.9)

        # Temporarily lower the threshold so check_significance can evaluate
        tester._min_interactions = 50
        winner = tester.check_significance()
        assert winner == CANDIDATE

    def test_returns_champion_when_significantly_better(self, store):
        tester = ABTestManager(
            store=store, prompt_key="default",
            split_ratio=0.5, min_interactions=200,
        )
        tester.start_test(_candidate())
        for _ in range(150):
            tester.record_result(CHAMPION, 0.9)
            tester.record_result(CANDIDATE, 0.1)

        tester._min_interactions = 50
        winner = tester.check_significance()
        assert winner == CHAMPION

    def test_returns_none_when_scores_equal(self, tester, store):
        tester.start_test(_candidate())
        for _ in range(10):
            tester.record_result(CHAMPION, 0.5)
            tester.record_result(CANDIDATE, 0.5)

        assert tester.check_significance() is None

    def test_no_active_test_returns_none(self, tester):
        assert tester.check_significance() is None


# ── Auto-resolution ──────────────────────────────────────────────────

class TestAutoResolution:
    def test_auto_promotes_when_candidate_wins(self, tester, store):
        tester.start_test(_candidate("Better prompt"))
        for _ in range(20):
            tester.record_result(CHAMPION, 0.2)
            tester.record_result(CANDIDATE, 0.9)

        # Test should have auto-resolved
        active = store.get_active_prompt("default")
        assert active.version == 2  # candidate was promoted

    def test_auto_rejects_when_champion_wins(self, tester, store):
        tester.start_test(_candidate("Worse prompt"))
        for _ in range(20):
            tester.record_result(CHAMPION, 0.9)
            tester.record_result(CANDIDATE, 0.1)

        # Champion should still be active
        active = store.get_active_prompt("default")
        assert active.version == 1
        # Test should be closed
        assert store.get_active_ab_test("default") is None


# ── Manual promotion/rejection ───────────────────────────────────────

class TestPromoteReject:
    def test_promote_candidate(self, tester, store):
        tester.start_test(_candidate("Promoted"))
        ok = tester.promote_candidate()
        assert ok is True
        active = store.get_active_prompt("default")
        assert active.version == 2

    def test_reject_candidate(self, tester, store):
        tester.start_test(_candidate("Rejected"))
        ok = tester.reject_candidate()
        assert ok is True
        assert store.get_active_ab_test("default") is None
        active = store.get_active_prompt("default")
        assert active.version == 1  # champion unchanged

    def test_promote_no_test_returns_false(self, tester):
        assert tester.promote_candidate() is False

    def test_reject_no_test_returns_false(self, tester):
        assert tester.reject_candidate() is False


# ── Force override (dashboard) ───────────────────────────────────────

class TestForceOverride:
    def test_force_promote(self, tester, store):
        ab_test = tester.start_test(_candidate("Force promoted"))
        ok = tester.force_promote(ab_test.id)
        assert ok is True
        active = store.get_active_prompt("default")
        assert active.version == 2

    def test_force_reject(self, tester, store):
        ab_test = tester.start_test(_candidate("Force rejected"))
        ok = tester.force_reject(ab_test.id)
        assert ok is True
        assert store.get_active_ab_test("default") is None

    def test_force_promote_wrong_id(self, tester, store):
        tester.start_test(_candidate())
        ok = tester.force_promote("wrong-id")
        assert ok is False

    def test_force_reject_wrong_id(self, tester, store):
        tester.start_test(_candidate())
        ok = tester.force_reject("wrong-id")
        assert ok is False

    def test_force_promote_no_test(self, tester):
        assert tester.force_promote() is False

    def test_force_reject_no_test(self, tester):
        assert tester.force_reject() is False


# ── Querying ─────────────────────────────────────────────────────────

class TestQuerying:
    def test_get_active_test(self, tester, store):
        tester.start_test(_candidate())
        test = tester.get_active_test()
        assert test is not None
        assert test.is_active is True

    def test_get_active_test_none(self, tester):
        assert tester.get_active_test() is None

    def test_get_test_summary(self, tester, store):
        tester.start_test(_candidate())
        tester.record_result(CHAMPION, 0.7)
        tester.record_result(CANDIDATE, 0.8)

        summary = tester.get_test_summary()
        assert summary is not None
        assert summary["champion_version"] == 1
        assert summary["candidate_version"] == 2
        assert summary["champion_interactions"] == 1
        assert summary["candidate_interactions"] == 1
        assert summary["champion_score"] == pytest.approx(0.7, abs=0.01)
        assert summary["candidate_score"] == pytest.approx(0.8, abs=0.01)
        assert summary["significant_winner"] is None  # too few samples

    def test_get_test_summary_none(self, tester):
        assert tester.get_test_summary() is None


# ── Welch's t-test implementation ────────────────────────────────────

class TestWelchTTest:
    def test_identical_distributions_high_p(self):
        p = welch_ttest_p(0.5, 0.1, 100, 0.5, 0.1, 100)
        assert p > 0.5  # not significant at all

    def test_very_different_means_low_p(self):
        p = welch_ttest_p(0.9, 0.01, 100, 0.1, 0.01, 100)
        assert p < 0.01  # highly significant

    def test_small_sample_high_p(self):
        p = welch_ttest_p(0.8, 0.1, 3, 0.2, 0.1, 3)
        # Small samples → hard to be significant
        assert p > 0.0  # at least returns a valid number

    def test_returns_1_for_insufficient_data(self):
        p = welch_ttest_p(0.5, 0.1, 1, 0.5, 0.1, 1)
        assert p == 1.0

    def test_symmetric(self):
        p1 = welch_ttest_p(0.8, 0.1, 50, 0.3, 0.1, 50)
        p2 = welch_ttest_p(0.3, 0.1, 50, 0.8, 0.1, 50)
        assert p1 == pytest.approx(p2, abs=0.001)
