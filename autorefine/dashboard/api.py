"""REST API logic for the AutoRefine dashboard.

Each method returns a plain dict/list ready for JSON serialisation.
The server module maps these to FastAPI routes.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from autorefine.ab_testing import ABTestManager
from autorefine.analytics import Analytics
from autorefine.cost_tracker import CostTracker
from autorefine.storage.base import BaseStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DashboardAPI:
    """Data layer for every dashboard endpoint."""

    def __init__(self, store: BaseStore, prompt_key: str = "default") -> None:
        self._store = store
        self._prompt_key = prompt_key
        self._analytics = Analytics(store, prompt_key)
        self._cost_tracker = CostTracker(store)
        self._ab = ABTestManager(store, prompt_key)

    # ── GET /api/prompts ─────────────────────────────────────────────

    def get_prompts(self) -> list[dict[str, Any]]:
        """List all prompt_keys with their active version info."""
        history = self._store.get_prompt_history(self._prompt_key)
        active = self._store.get_active_prompt(self._prompt_key)
        return {
            "prompt_key": self._prompt_key,
            "active_version": active.version if active else 0,
            "active_prompt": active.system_prompt if active else "",
            "total_versions": len(history),
            "versions": [
                {
                    "version": pv.version,
                    "changelog": pv.changelog,
                    "is_active": pv.is_active,
                    "created_at": pv.created_at.isoformat(),
                    "performance_score": round(pv.performance_score, 4),
                }
                for pv in history
            ],
        }

    # ── GET /api/prompts/{key}/history ────────────────────────────────

    def get_prompt_history(self, prompt_key: str = "") -> list[dict[str, Any]]:
        """Version history with diffs between consecutive versions."""
        key = prompt_key or self._prompt_key
        history = self._store.get_prompt_history(key)
        result = []
        prev_text = ""
        for pv in history:
            diff = ""
            if prev_text:
                diff = "\n".join(difflib.unified_diff(
                    prev_text.splitlines(), pv.system_prompt.splitlines(),
                    fromfile=f"v{pv.parent_version or pv.version - 1}",
                    tofile=f"v{pv.version}", lineterm="",
                ))
            result.append({
                "version": pv.version,
                "system_prompt": pv.system_prompt,
                "parent_version": pv.parent_version,
                "changelog": pv.changelog,
                "is_active": pv.is_active,
                "created_at": pv.created_at.isoformat(),
                "diff": diff,
            })
            prev_text = pv.system_prompt
        return result

    # ── POST /api/prompts/{key}/rollback/{version} ────────────────────

    def rollback(self, version: int, prompt_key: str = "") -> dict[str, Any]:
        key = prompt_key or self._prompt_key
        pv = self._store.get_prompt_version(key, version)
        if pv is None:
            return {"error": f"Version {version} not found", "status": "error"}
        self._store.set_active_version(key, version)
        return {"status": "ok", "active_version": version}

    # ── GET /api/analytics ───────────────────────────────────────────

    def get_analytics(self, days: int = 30, prompt_key: str = "") -> dict[str, Any]:
        """Improvement curves — score per version and per day."""
        key = prompt_key or self._prompt_key
        analytics = Analytics(self._store, key)
        snap = analytics.snapshot(days)

        # Build daily score aggregation
        since = _utc_now() - timedelta(days=days)
        feedback = self._store.get_feedback(prompt_key=key, limit=100_000, since=since)
        daily: dict[str, list[float]] = defaultdict(list)
        for fb in feedback:
            day = fb.created_at.strftime("%Y-%m-%d")
            daily[day].append(fb.score)

        daily_scores = [
            {"date": d, "avg_score": round(sum(s) / len(s), 4), "count": len(s)}
            for d, s in sorted(daily.items())
        ]

        return {
            "prompt_key": key,
            "days": days,
            "total_interactions": snap.total_interactions,
            "total_feedback": snap.total_feedback,
            "average_score": round(snap.average_score, 4),
            "positive_rate": round(snap.positive_rate, 4),
            "negative_rate": round(snap.negative_rate, 4),
            "active_version": snap.active_version,
            "refiner_cost": round(snap.refiner_cost, 4),
            "feedback_distribution": snap.feedback_distribution,
            "improvement_curve": snap.improvement_curve,
            "daily_scores": daily_scores,
            "score_by_version": {
                str(k): round(v, 4) for k, v in snap.score_by_version.items()
            },
        }

    # ── GET /api/feedback ────────────────────────────────────────────

    def get_feedback(
        self,
        limit: int = 50,
        prompt_key: str = "",
        signal_type: str = "",
        since_days: int = 0,
    ) -> dict[str, Any]:
        """Paginated recent feedback with filters."""
        key = prompt_key or self._prompt_key
        since = _utc_now() - timedelta(days=since_days) if since_days > 0 else None
        items = self._store.get_feedback(prompt_key=key, limit=limit, since=since)

        if signal_type:
            items = [fb for fb in items if fb.feedback_type.value == signal_type]

        return {
            "prompt_key": key,
            "total": len(items),
            "items": [
                {
                    "id": fb.id,
                    "interaction_id": fb.interaction_id,
                    "type": fb.feedback_type.value,
                    "score": fb.score,
                    "confidence": fb.confidence,
                    "comment": fb.comment,
                    "correction": fb.correction,
                    "user_id": fb.user_id,
                    "created_at": fb.created_at.isoformat(),
                }
                for fb in items
            ],
        }

    # ── GET /api/ab-tests ────────────────────────────────────────────

    def get_ab_tests(self, prompt_key: str = "") -> dict[str, Any]:
        key = prompt_key or self._prompt_key
        test = self._store.get_active_ab_test(key)
        if test is None:
            return {"active": False, "test": None}
        return {
            "active": True,
            "test": {
                "id": test.id,
                "prompt_key": test.prompt_key,
                "champion_version": test.control_version,
                "candidate_version": test.candidate_version,
                "split_ratio": test.split_ratio,
                "champion_interactions": test.control_interactions,
                "candidate_interactions": test.candidate_interactions,
                "champion_score": round(test.control_score, 4),
                "candidate_score": round(test.candidate_score, 4),
                "created_at": test.created_at.isoformat(),
            },
        }

    # ── POST /api/ab-tests/{key}/promote & /reject ───────────────────

    def promote_ab_test(self, test_id: str = "") -> dict[str, Any]:
        ok = self._ab.force_promote(test_id or None)
        return {"status": "promoted" if ok else "not_found"}

    def reject_ab_test(self, test_id: str = "") -> dict[str, Any]:
        ok = self._ab.force_reject(test_id or None)
        return {"status": "rejected" if ok else "not_found"}

    # ── GET /api/costs ───────────────────────────────────────────────

    def get_costs(self) -> dict[str, Any]:
        summary = self._cost_tracker.summary()
        return summary

    # ── Cloud usage methods (optional — no-op without cloud billing) ──

    def get_usage_summary(self, org_id: str = "") -> dict[str, Any]:
        """Current month's usage summary for the billing panel."""
        return {
            "org_id": org_id,
            "monthly_spend": 0.0,
            "monthly_cap": 0.0,
            "utilization_pct": 0.0,
            "cloud_mode": False,
        }

    def get_daily_usage(self, org_id: str = "", days: int = 30) -> list[dict[str, Any]]:
        """Daily usage breakdown for the spend chart."""
        return []

    def get_model_breakdown(self, org_id: str = "", days: int = 30) -> dict[str, Any]:
        """Usage breakdown by model for the pie chart."""
        return {"models": {}, "total_calls": 0}
