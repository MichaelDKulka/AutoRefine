"""Public API for AutoRefine — the only file most developers need to read."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import Any, Callable

from autorefine.ab_testing import ABTestManager
from autorefine.analytics import Analytics, AnalyticsSnapshot
from autorefine.config import AutoRefineSettings
from autorefine.cost_tracker import CostTracker
from autorefine.exceptions import RollbackError
from autorefine.feedback import FeedbackCollector
from autorefine.feedback_filter import FeedbackFilter
from autorefine.feedback_provider import FeedbackProvider
from autorefine.interceptor import Interceptor
from autorefine.models import CompletionResponse, FeedbackSignal, Message, PromptVersion
from autorefine.notifications import PromptChangeEvent, PromptChangeNotifier
from autorefine.pii_scrubber import PIIScrubber
from autorefine.providers import get_provider
from autorefine.refiner import Refiner
from autorefine.storage import get_store
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine")


class AutoRefine:
    """Drop-in LLM client that makes your AI progressively smarter."""

    def __init__(self, api_key: str = "", model: str = "gpt-4o",
                 refiner_key: str = "", refiner_model: str = "",
                 auto_learn: bool = False, prompt_key: str = "default",
                 store: BaseStore | None = None,
                 on_refine: Callable | None = None,
                 on_prompt_change: Callable | None = None,
                 feedback_provider: FeedbackProvider | None = None,
                 **config_overrides: Any) -> None:
        cfg = AutoRefineSettings(
            api_key=api_key, model=model, refiner_key=refiner_key,
            auto_learn=auto_learn,
            **{k: v for k, v in config_overrides.items() if v is not None})
        if api_key:
            cfg.api_key = api_key
        if refiner_key:
            cfg.refiner_key = refiner_key
        if refiner_model:
            cfg.refiner_model = refiner_model
        if auto_learn:
            cfg.auto_learn = auto_learn
        self._cfg, self._prompt_key = cfg, prompt_key

        self._store = store or get_store(config=cfg)
        self._provider = get_provider(cfg.detect_provider(), api_key=cfg.api_key, model=cfg.model)
        self._interceptor = Interceptor(provider=self._provider, store=self._store, prompt_key=prompt_key)
        self._cost_tracker = CostTracker(self._store, cfg.cost_limit_monthly)
        scrubber = PIIScrubber(enabled=cfg.pii_scrub_enabled)
        fb_filter = FeedbackFilter(enabled=cfg.feedback_filter_enabled)
        self._notifier = PromptChangeNotifier(webhook_url=cfg.webhook_url, on_prompt_change=on_prompt_change)

        self._refiner: Refiner | None = None
        self._ab: ABTestManager | None = None
        if cfg.refiner_key:
            self._refiner = Refiner(
                refiner_provider=get_provider(cfg.refiner_provider, api_key=cfg.refiner_key, model=cfg.refiner_model),
                store=self._store, prompt_key=prompt_key, batch_size=cfg.refine_batch_size,
                cost_limit=cfg.cost_limit_monthly, pii_scrubber=scrubber, feedback_filter=fb_filter)
            self._ab = ABTestManager(
                store=self._store, prompt_key=prompt_key,
                split_ratio=cfg.ab_test_split, min_interactions=cfg.ab_test_min_interactions)

        self._feedback = FeedbackCollector(
            store=self._store, prompt_key=prompt_key, refine_threshold=cfg.refine_threshold,
            on_ready=self._run_refinement if cfg.auto_learn else None)
        self._analytics = Analytics(self._store, prompt_key)
        self._on_refine = on_refine
        self._feedback_provider = feedback_provider

    # ── LLM calls ────────────────────────────────────────────────────

    def chat(self, system: str, messages: list[dict | Message],
             prompt_key: str = "", **kw: Any) -> CompletionResponse:
        """Multi-turn chat completion."""
        return self._interceptor.intercept_call(
            prompt_key=prompt_key or self._prompt_key,
            system_prompt=system, messages_or_prompt=messages, call_type="chat", **kw)

    def complete(self, system: str, prompt: str,
                 prompt_key: str = "", **kw: Any) -> CompletionResponse:
        """Single-prompt completion."""
        return self._interceptor.intercept_call(
            prompt_key=prompt_key or self._prompt_key,
            system_prompt=system, messages_or_prompt=prompt, call_type="complete", **kw)

    def stream(self, system: str, messages: list[dict | Message],
               prompt_key: str = "", **kw: Any) -> Iterator[str]:
        """Streaming chat completion."""
        return self._interceptor.intercept_call(
            prompt_key=prompt_key or self._prompt_key,
            system_prompt=system, messages_or_prompt=messages, call_type="stream", **kw)

    # ── Feedback ─────────────────────────────────────────────────────

    def feedback(self, response_id: str, signal: str,
                 comment: str | None = None, **kw: Any) -> FeedbackSignal:
        """Record feedback for a response."""
        return self._feedback.submit(
            interaction_id=response_id, signal=signal, comment=comment or "", **kw)

    def collect_feedback(self, response: CompletionResponse) -> FeedbackSignal | None:
        """Collect feedback using the configured :class:`FeedbackProvider`.

        Calls the provider's :meth:`~FeedbackProvider.get_feedback` to obtain
        a feedback string from the end-user, classifies it as positive or
        negative, and records it.  Returns ``None`` if no provider is
        configured or the user provided an empty string (skipped).

        Args:
            response: The :class:`CompletionResponse` returned by
                :meth:`chat`, :meth:`complete`, or :meth:`stream`.

        Returns:
            The recorded :class:`FeedbackSignal`, or ``None`` if skipped.
        """
        if self._feedback_provider is None:
            logger.debug("No feedback provider configured — skipping collection")
            return None

        text = self._feedback_provider.get_feedback(response.id, response.text)
        if not text or not text.strip():
            return None

        signal = self._feedback_provider.classify(text)
        return self._feedback.submit(
            interaction_id=response.id, signal=signal, comment=text)

    # ── Prompt management ────────────────────────────────────────────

    def get_active_prompt(self, prompt_key: str = "") -> PromptVersion | None:
        return self._store.get_active_prompt(prompt_key or self._prompt_key)

    def get_prompt_history(self, prompt_key: str = "") -> list[PromptVersion]:
        return self._store.get_prompt_history(prompt_key or self._prompt_key)

    def set_system_prompt(self, prompt: str, prompt_key: str = "") -> PromptVersion:
        """Manually set a system prompt (creates a new version)."""
        key = prompt_key or self._prompt_key
        history = self._store.get_prompt_history(key)
        next_v = max((pv.version for pv in history), default=0) + 1
        pv = PromptVersion(version=next_v, prompt_key=key, system_prompt=prompt,
                           changelog="initial prompt" if next_v == 1 else "manually set", is_active=True)
        self._store.set_active_version(key, -1)
        self._store.save_prompt_version(pv)
        return pv

    def rollback(self, version: int, prompt_key: str = "") -> PromptVersion:
        """Roll back to a specific prompt version."""
        key = prompt_key or self._prompt_key
        pv = self._store.get_prompt_version(key, version)
        if pv is None:
            raise RollbackError(f"Version {version} not found", version=version)
        self._store.set_active_version(key, version)
        return pv

    def refine_now(self) -> PromptVersion | None:
        """Manually trigger a refinement cycle."""
        return self._run_refinement()

    # ── Analytics & costs ────────────────────────────────────────────

    @property
    def analytics(self) -> Analytics:
        return self._analytics

    def get_analytics(self, days: int = 30) -> AnalyticsSnapshot:
        return self._analytics.snapshot(days=days)

    @property
    def costs(self) -> dict:
        return self._cost_tracker.summary()

    # ── Dashboard ────────────────────────────────────────────────────

    def start_dashboard(self, port: int | None = None) -> threading.Thread:
        """Launch the analytics dashboard in a background thread."""
        from autorefine.dashboard.server import run_dashboard
        p = port or self._cfg.dashboard_port
        t = threading.Thread(target=run_dashboard, daemon=True, name="autorefine-dashboard",
                             args=(self._store, self._prompt_key, p, self._cfg.dashboard_password,
                                   self._cfg.cors_origins, self._cfg.dashboard_rate_limit))
        t.start()
        logger.info("Dashboard started on http://localhost:%d", p)
        return t

    # ── Widget ────────────────────────────────────────────────────────

    def get_widget_html(self, response_id: str, style: str = "minimal",
                        endpoint: str = "") -> str:
        """Return an embeddable HTML feedback widget for a response.

        Args:
            response_id: The ``.id`` from a chat/complete response.
            style: ``"minimal"``, ``"standard"``, or ``"detailed"``.
            endpoint: Base URL for the feedback POST. Defaults to
                ``http://localhost:{dashboard_port}``.
        """
        from autorefine.widget import FeedbackWidget
        ep = endpoint or f"http://localhost:{self._cfg.dashboard_port}"
        return FeedbackWidget(endpoint=ep).render(response_id, style=style)

    # ── GDPR compliance ─────────────────────────────────────────────

    def export_data(self, prompt_key: str = "") -> dict[str, Any]:
        """Export all data for a prompt_key (GDPR data portability).

        Returns a dict with all interactions, feedback, prompt versions,
        and cost entries associated with the given prompt_key.
        """
        key = prompt_key or self._prompt_key
        return {
            "prompt_key": key,
            "interactions": [ix.model_dump(mode="json") for ix in
                             self._store.get_interactions(key, limit=100_000)],
            "feedback": [fb.model_dump(mode="json") for fb in
                         self._store.get_feedback(key, limit=100_000)],
            "prompt_versions": [pv.model_dump(mode="json") for pv in
                                self._store.get_prompt_history(key)],
        }

    def delete_data(self, prompt_key: str = "") -> dict[str, int]:
        """Delete all data for a prompt_key (GDPR right to deletion).

        Returns counts of deleted records.
        """
        from datetime import datetime, timedelta, timezone
        key = prompt_key or self._prompt_key
        # Count before
        interactions = len(self._store.get_interactions(key, limit=100_000))
        feedback = len(self._store.get_feedback(key, limit=100_000))
        versions = len(self._store.get_prompt_history(key))
        # Purge everything (use far-future cutoff)
        self._store.purge_old_data(datetime.now(timezone.utc) + timedelta(days=1))
        self._store.set_active_version(key, -1)
        logger.info("Deleted data for prompt_key=%s: %d interactions, %d feedback, %d versions",
                     key, interactions, feedback, versions)
        return {"interactions": interactions, "feedback": feedback, "versions": versions}

    # ── Health check ────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """Verify provider, refiner, and store are all reachable.

        Returns a status dict with ``ok`` (bool) and per-component status.
        """
        result: dict[str, Any] = {"ok": True, "provider": "unknown", "store": "unknown", "refiner": "unknown"}

        # Check store
        try:
            self._store.get_active_prompt(self._prompt_key)
            result["store"] = "ok"
        except Exception as exc:
            result["store"] = f"error: {exc}"
            result["ok"] = False

        # Check primary provider (zero-cost ping)
        try:
            from autorefine.models import Message, MessageRole
            self._provider.chat("", [Message(role=MessageRole.USER, content="ping")], max_tokens=1)
            result["provider"] = "ok"
        except Exception as exc:
            result["provider"] = f"error: {exc}"
            result["ok"] = False

        # Check refiner
        if self._refiner is None:
            result["refiner"] = "not_configured"
        else:
            try:
                self._refiner._provider.chat("", [Message(role=MessageRole.USER, content="ping")], max_tokens=1)
                result["refiner"] = "ok"
            except Exception as exc:
                result["refiner"] = f"error: {exc}"
                result["ok"] = False

        try:
            result["monthly_refiner_cost"] = self._cost_tracker.summary().get("monthly_refiner_spend", 0)
        except Exception:
            result["monthly_refiner_cost"] = -1
        return result

    # ── Internal ─────────────────────────────────────────────────────

    def _run_refinement(self) -> PromptVersion | None:
        if self._refiner is None:
            logger.warning("No refiner configured — skipping")
            return None
        try:
            candidate = self._refiner.refine()
        except Exception:
            logger.error("Refinement failed", exc_info=True)
            return None
        if self._ab and self._cfg.ab_test_split > 0:
            self._ab.start_test(candidate)
            if self._on_refine:
                self._on_refine(candidate)
            return None
        current = self._store.get_active_prompt(self._prompt_key)
        version = self._refiner.promote_candidate(candidate)
        self._notifier.notify(PromptChangeEvent(
            prompt_key=self._prompt_key,
            old_version=current.version if current else 0,
            new_version=version.version, changelog=candidate.changelog,
            trigger="auto_refine",
            old_prompt=current.system_prompt if current else "",
            new_prompt=version.system_prompt))
        if self._on_refine:
            self._on_refine(version)
        return version
