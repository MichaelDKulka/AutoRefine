"""Async public API for AutoRefine — mirrors the sync AutoRefine client.

Usage::

    from autorefine import AsyncAutoRefine

    client = AsyncAutoRefine(api_key="sk-...", model="gpt-4o",
                             refiner_key="sk-ant-...", auto_learn=True)

    resp = await client.chat("Be helpful.", [{"role": "user", "content": "Hi"}])
    await client.feedback(resp.id, "thumbs_up")
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Callable

from autorefine.ab_testing import ABTestManager
from autorefine.analytics import Analytics, AnalyticsSnapshot
from autorefine.config import AutoRefineSettings
from autorefine.cost_tracker import CostTracker
from autorefine.exceptions import RollbackError
from autorefine.feedback import FeedbackCollector
from autorefine.feedback_filter import FeedbackFilter
from autorefine.dimensions import FeedbackDimensionSchema
from autorefine.directives import DirectiveManager, RefinementDirectives
from autorefine.feedback_provider import FeedbackProvider
from autorefine.interceptor import Interceptor
from autorefine.models import CompletionResponse, FeedbackSignal, FeedbackType, Message, PromptVersion
from autorefine.outcomes import OutcomeTranslator
from autorefine.notifications import PromptChangeEvent, PromptChangeNotifier
from autorefine.pii_scrubber import PIIScrubber
from autorefine.providers import get_provider
from autorefine.refiner import Refiner
from autorefine.storage import get_store
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.async")


class AsyncAutoRefine:
    """Async drop-in LLM client that makes your AI progressively smarter.

    All public methods are ``async``.  Uses the provider's native async
    methods (``async_chat``, ``async_stream``) when available, falling
    back to ``asyncio.to_thread`` for providers that only have sync.

    The constructor is synchronous — only LLM calls, feedback, and
    store operations are async.
    """

    def __init__(
        self, api_key: str = "", model: str = "gpt-4o",
        refiner_key: str = "", refiner_model: str = "",
        auto_learn: bool = False, prompt_key: str = "default",
        store: BaseStore | None = None,
        on_refine: Callable | None = None,
        on_prompt_change: Callable | None = None,
        feedback_provider: FeedbackProvider | None = None,
        feedback_dimensions: dict[str, dict[str, Any]] | None = None,
        **config_overrides: Any,
    ) -> None:
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
        self._cloud_mode = cfg.detect_cloud_mode()

        if self._cloud_mode:
            self._provider = get_provider(
                api_key=cfg.api_key, model=cfg.model,
                base_url=cfg.cloud_base_url, timeout=cfg.cloud_timeout,
            )
            logger.info("AutoRefine Cloud mode: calls route through %s", cfg.cloud_base_url)
        else:
            self._provider = get_provider(cfg.detect_provider(), api_key=cfg.api_key, model=cfg.model)

        self._interceptor = Interceptor(provider=self._provider, store=self._store, prompt_key=prompt_key)
        self._cost_tracker = CostTracker(self._store, cfg.cost_limit_monthly)
        scrubber = PIIScrubber(enabled=cfg.pii_scrub_enabled)
        fb_filter = FeedbackFilter(enabled=cfg.feedback_filter_enabled)
        self._notifier = PromptChangeNotifier(webhook_url=cfg.webhook_url, on_prompt_change=on_prompt_change)

        # ── Dimensions ──
        self._dimension_schema: FeedbackDimensionSchema | None = None
        if feedback_dimensions:
            self._dimension_schema = FeedbackDimensionSchema.from_dict(
                prompt_key, feedback_dimensions
            )
            self._store.save_dimension_schema(self._dimension_schema)

        # ── Directives manager ──
        self._directive_manager = DirectiveManager(self._store)

        # ── Outcome translator ──
        self._outcome_translator = OutcomeTranslator(
            dimension_schema=self._dimension_schema
        )

        self._refiner: Refiner | None = None
        self._ab: ABTestManager | None = None
        if self._cloud_mode:
            pass
        elif cfg.refiner_key:
            self._refiner = Refiner(
                refiner_provider=get_provider(cfg.refiner_provider, api_key=cfg.refiner_key, model=cfg.refiner_model),
                store=self._store, prompt_key=prompt_key, batch_size=cfg.refine_batch_size,
                cost_limit=cfg.cost_limit_monthly, pii_scrubber=scrubber, feedback_filter=fb_filter,
                dimension_schema=self._dimension_schema,
                directive_manager=self._directive_manager)
            self._ab = ABTestManager(
                store=self._store, prompt_key=prompt_key,
                split_ratio=cfg.ab_test_split, min_interactions=cfg.ab_test_min_interactions)

        self._feedback = FeedbackCollector(
            store=self._store, prompt_key=prompt_key, refine_threshold=cfg.refine_threshold,
            on_ready=self._sync_run_refinement if cfg.auto_learn else None,
            dimension_schema=self._dimension_schema)
        self._analytics = Analytics(self._store, prompt_key)
        self._on_refine = on_refine
        self._feedback_provider = feedback_provider

    # ── Async LLM calls ──────────────────────────────────────────────

    async def chat(self, system: str, messages: list[dict | Message],
                   prompt_key: str = "", **kw: Any) -> CompletionResponse:
        """Async multi-turn chat completion."""
        return await self._interceptor.async_intercept_call(
            prompt_key=prompt_key or self._prompt_key,
            system_prompt=system, messages_or_prompt=messages, call_type="chat", **kw)

    async def complete(self, system: str, prompt: str,
                       prompt_key: str = "", **kw: Any) -> CompletionResponse:
        """Async single-prompt completion."""
        return await self._interceptor.async_intercept_call(
            prompt_key=prompt_key or self._prompt_key,
            system_prompt=system, messages_or_prompt=prompt, call_type="complete", **kw)

    async def stream(self, system: str, messages: list[dict | Message],
                     prompt_key: str = "", **kw: Any) -> AsyncIterator[str]:
        """Async streaming chat completion."""
        return await self._interceptor.async_intercept_call(
            prompt_key=prompt_key or self._prompt_key,
            system_prompt=system, messages_or_prompt=messages, call_type="stream", **kw)

    # ── Async feedback ───────────────────────────────────────────────

    async def feedback(self, response_id: str, signal: str = "",
                       comment: str | None = None,
                       dimensions: dict[str, float] | None = None,
                       context: dict[str, Any] | None = None,
                       **kw: Any) -> FeedbackSignal:
        """Async feedback recording with optional dimensions/context."""
        if not signal and dimensions:
            if self._dimension_schema:
                composite = self._dimension_schema.compute_composite(dimensions)
            else:
                composite = sum(dimensions.values()) / len(dimensions)
            signal = "positive" if composite >= 0 else "negative"

        return await self._feedback.async_submit(
            interaction_id=response_id, signal=signal or "positive",
            comment=comment or "", dimensions=dimensions, context=context, **kw)

    async def collect_feedback(self, response: CompletionResponse) -> FeedbackSignal | None:
        """Collect feedback using the configured :class:`FeedbackProvider`.

        Calls the provider's :meth:`~FeedbackProvider.get_feedback` in a
        thread (since providers may block on I/O), classifies the result,
        and records it.  Returns ``None`` if no provider is configured or
        the user provided an empty string.

        Args:
            response: The :class:`CompletionResponse` from a chat/complete call.

        Returns:
            The recorded :class:`FeedbackSignal`, or ``None`` if skipped.
        """
        if self._feedback_provider is None:
            return None

        text = await asyncio.to_thread(
            self._feedback_provider.get_feedback, response.id, response.text)
        if not text or not text.strip():
            return None

        signal = self._feedback_provider.classify(text)
        return await self._feedback.async_submit(
            interaction_id=response.id, signal=signal, comment=text)

    # ── Refinement directives ────────────────────────────────────────

    async def set_refinement_directives(
        self, prompt_key: str = "",
        directives: list[str] | None = None,
        domain_context: str | None = None,
        preserve_behaviors: list[str] | None = None,
    ) -> RefinementDirectives:
        key = prompt_key or self._prompt_key
        return await asyncio.to_thread(
            self._directive_manager.set, key,
            directives=directives, domain_context=domain_context,
            preserve_behaviors=preserve_behaviors,
        )

    async def update_refinement_directives(
        self, prompt_key: str = "",
        add_directives: list[str] | None = None,
        remove_directives: list[str] | None = None,
        domain_context: str | None = None,
        preserve_behaviors: list[str] | None = None,
    ) -> RefinementDirectives:
        key = prompt_key or self._prompt_key
        return await asyncio.to_thread(
            self._directive_manager.update, key,
            add_directives=add_directives,
            remove_directives=remove_directives,
            domain_context=domain_context,
            preserve_behaviors=preserve_behaviors,
        )

    # ── Outcome-based feedback ────────────────────────────────────────

    async def report_outcome(
        self, response_id: str, outcome: dict[str, Any],
        dimension_overrides: dict[str, float] | None = None,
        context: dict[str, Any] | None = None,
    ) -> FeedbackSignal | None:
        """Async report_outcome."""
        def _sync() -> FeedbackSignal | None:
            ix = self._store.get_interaction(response_id)
            if ix is None:
                return None
            dim_scores = self._outcome_translator.translate(
                outcome, dimension_overrides=dimension_overrides, interaction=ix)
            fb_context = dict(context or {})
            fb_context.update({k: v for k, v in outcome.items()
                               if k in ("predicted", "actual", "correct")})
            correct = outcome.get("correct", False)
            signal = "positive" if correct else "negative"
            return self._feedback.submit(
                interaction_id=response_id, signal=signal,
                dimensions=dim_scores, context=fb_context,
                metadata={"feedback_source": "outcome"})
        return await asyncio.to_thread(_sync)

    # ── Prompt management (thin async wrappers) ──────────────────────

    async def get_active_prompt(self, prompt_key: str = "") -> PromptVersion | None:
        return await asyncio.to_thread(self._store.get_active_prompt, prompt_key or self._prompt_key)

    async def get_prompt_history(self, prompt_key: str = "") -> list[PromptVersion]:
        return await asyncio.to_thread(self._store.get_prompt_history, prompt_key or self._prompt_key)

    async def set_system_prompt(self, prompt: str, prompt_key: str = "") -> PromptVersion:
        def _sync():
            key = prompt_key or self._prompt_key
            history = self._store.get_prompt_history(key)
            next_v = max((pv.version for pv in history), default=0) + 1
            pv = PromptVersion(version=next_v, prompt_key=key, system_prompt=prompt,
                               changelog="initial prompt" if next_v == 1 else "manually set", is_active=True)
            self._store.set_active_version(key, -1)
            self._store.save_prompt_version(pv)
            return pv
        return await asyncio.to_thread(_sync)

    async def rollback(self, version: int, prompt_key: str = "") -> PromptVersion:
        def _sync():
            key = prompt_key or self._prompt_key
            pv = self._store.get_prompt_version(key, version)
            if pv is None:
                raise RollbackError(f"Version {version} not found", version=version)
            self._store.set_active_version(key, version)
            return pv
        return await asyncio.to_thread(_sync)

    async def refine_now(self) -> PromptVersion | None:
        return await asyncio.to_thread(self._sync_run_refinement)

    # ── Analytics ────────────────────────────────────────────────────

    @property
    def analytics(self) -> Analytics:
        return self._analytics

    async def get_analytics(self, days: int = 30) -> AnalyticsSnapshot:
        return await asyncio.to_thread(self._analytics.snapshot, days)

    @property
    def costs(self) -> dict:
        return self._cost_tracker.summary()

    # ── Health check ─────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Async health check — verifies provider, refiner, and store."""
        from autorefine.models import MessageRole
        result: dict[str, Any] = {"ok": True, "provider": "unknown", "store": "unknown", "refiner": "unknown"}
        try:
            await asyncio.to_thread(self._store.get_active_prompt, self._prompt_key)
            result["store"] = "ok"
        except Exception as exc:
            result["store"] = f"error: {exc}"
            result["ok"] = False
        try:
            await self._provider.async_chat("", [Message(role=MessageRole.USER, content="ping")], max_tokens=1)
            result["provider"] = "ok"
        except Exception as exc:
            result["provider"] = f"error: {exc}"
            result["ok"] = False
        if self._refiner is None:
            result["refiner"] = "not_configured"
        else:
            try:
                await self._refiner._provider.async_chat("", [Message(role=MessageRole.USER, content="ping")], max_tokens=1)
                result["refiner"] = "ok"
            except Exception as exc:
                result["refiner"] = f"error: {exc}"
                result["ok"] = False
        return result

    # ── Internal ─────────────────────────────────────────────────────

    def _sync_run_refinement(self) -> PromptVersion | None:
        """Sync refinement runner (called from feedback threshold callback)."""
        if self._refiner is None:
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
