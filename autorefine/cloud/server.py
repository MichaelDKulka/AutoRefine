"""FastAPI application for AutoRefine Cloud API.

Serves two sets of endpoints:
1. Proxy API (/v1/*) -- handles proxied LLM calls from the SDK
2. Management API (/api/*) -- handles key management, billing, dashboard

Usage::

    from autorefine.cloud.server import create_cloud_app

    app = create_cloud_app(store, authenticator, proxy, billing)
    # Run with: uvicorn autorefine.cloud.server:app
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any

from autorefine.cloud.auth import Authenticator
from autorefine.cloud.billing import BillingManager
from autorefine.cloud.proxy import LLMProxy
from autorefine.exceptions import (
    AutoRefineError,
    CloudAuthError,
    ProviderRateLimitError,
    SpendCapExceeded,
)
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.cloud.server")


def create_cloud_app(
    store: BaseStore,
    authenticator: Authenticator,
    proxy: LLMProxy,
    billing: BillingManager,
    cors_origins: str = "*",
) -> Any:
    """Create the FastAPI application for AutoRefine Cloud.

    Returns None if FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError:
        logger.error("FastAPI not installed: pip install autorefine[dashboard]")
        return None

    app = FastAPI(title="AutoRefine Cloud", docs_url=None, redoc_url=None)

    # ── Auth middleware ────────────────────────────────────────────────

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next: Any) -> Any:
        path = request.url.path

        # Skip auth for health check and OPTIONS
        if path == "/health" or request.method == "OPTIONS":
            return await call_next(request)

        # Extract Bearer token
        if path.startswith("/v1/") or path.startswith("/api/"):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Missing Authorization: Bearer <api_key>"},
                )
            raw_key = auth_header[7:]
            try:
                org, api_key = authenticator.validate(raw_key)
                request.state.org = org
                request.state.api_key = api_key
            except CloudAuthError as exc:
                return JSONResponse(status_code=401, content={"error": str(exc)})
            except SpendCapExceeded as exc:
                return JSONResponse(status_code=402, content={"error": str(exc)})
            except ProviderRateLimitError as exc:
                return JSONResponse(status_code=429, content={"error": str(exc)})

        return await call_next(request)

    # ── CORS ──────────────────────────────────────────────────────────

    origins = [o.strip() for o in cors_origins.split(",") if o.strip()] if cors_origins else []

    @app.middleware("http")
    async def cors_middleware(request: Request, call_next: Any) -> Any:
        origin = request.headers.get("Origin", "")
        if request.method == "OPTIONS":
            headers = {
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            }
            if "*" in origins or origin in origins:
                headers["Access-Control-Allow-Origin"] = origin or "*"
            return JSONResponse(content={}, headers=headers)
        response = await call_next(request)
        if origin and ("*" in origins or origin in origins):
            response.headers["Access-Control-Allow-Origin"] = origin
        return response

    # ── Health check ──────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "autorefine-cloud"}

    # ══════════════════════════════════════════════════════════════════
    # Proxy endpoints (/v1/*)
    # ══════════════════════════════════════════════════════════════════

    @app.post("/v1/chat")
    async def v1_chat(request: Request) -> JSONResponse:
        """Proxy a chat completion to the upstream provider."""
        org = request.state.org
        api_key = request.state.api_key
        data = await request.json()

        model = data.get("model", "gpt-4o")
        system_prompt = data.get("system_prompt", "")
        messages = data.get("messages", [])
        prompt_key = data.get("prompt_key", "default")

        # Extract kwargs (temperature, max_tokens, etc.)
        kwargs = {
            k: v for k, v in data.items()
            if k not in ("model", "system_prompt", "messages", "prompt_key", "api_key")
        }

        try:
            result = proxy.chat(
                org, api_key, model, system_prompt, messages,
                prompt_key=prompt_key, **kwargs,
            )
        except AutoRefineError as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        except Exception as exc:
            logger.error("Proxy chat failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=502, content={"error": "Upstream provider error"})

        return JSONResponse(content={
            "text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "model": result.model,
            "finish_reason": result.finish_reason,
            "interaction_id": result.interaction_id,
            "cost": {
                "upstream": round(result.upstream_cost, 6),
                "markup": round(result.markup_amount, 6),
                "total": round(result.customer_cost, 6),
            },
        })

    @app.post("/v1/chat/stream")
    async def v1_chat_stream(request: Request) -> StreamingResponse:
        """Proxy a streaming chat completion (NDJSON)."""
        org = request.state.org
        api_key = request.state.api_key
        data = await request.json()

        model = data.get("model", "gpt-4o")
        system_prompt = data.get("system_prompt", "")
        messages = data.get("messages", [])
        prompt_key = data.get("prompt_key", "default")

        kwargs = {
            k: v for k, v in data.items()
            if k not in ("model", "system_prompt", "messages", "prompt_key", "api_key")
        }

        def generate():
            try:
                for chunk in proxy.stream(
                    org, api_key, model, system_prompt, messages,
                    prompt_key=prompt_key, **kwargs,
                ):
                    yield _json.dumps(chunk) + "\n"
            except Exception as exc:
                logger.error("Stream proxy error: %s", exc, exc_info=True)
                yield _json.dumps({"error": str(exc), "done": True}) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
        )

    @app.post("/v1/feedback")
    async def v1_feedback(request: Request) -> JSONResponse:
        """Record feedback for an interaction."""
        from autorefine.feedback import FeedbackCollector

        org = request.state.org
        data = await request.json()

        interaction_id = data.get("interaction_id", "")
        signal = data.get("signal", "")
        comment = data.get("comment", "")
        dimensions = data.get("dimensions")
        context = data.get("context")

        if not interaction_id or not signal:
            return JSONResponse(
                status_code=400,
                content={"error": "interaction_id and signal are required"},
            )

        namespace = org.slug or org.id
        collector = FeedbackCollector(store=store, prompt_key=namespace)
        fb = collector.submit(
            interaction_id=interaction_id,
            signal=signal,
            comment=comment,
            dimensions=dimensions,
            context=context,
        )
        return JSONResponse(content={
            "feedback_id": fb.id,
            "score": fb.score,
        })

    # ══════════════════════════════════════════════════════════════════
    # Management endpoints (/api/*)
    # ══════════════════════════════════════════════════════════════════

    @app.get("/api/usage")
    async def api_usage(request: Request) -> JSONResponse:
        """Current month's usage summary."""
        org = request.state.org
        spend = billing.get_monthly_spend(org.id)
        cap = org.monthly_spend_cap
        return JSONResponse(content={
            "org_id": org.id,
            "monthly_spend": round(spend, 4),
            "monthly_cap": cap,
            "utilization_pct": round(spend / cap * 100, 1) if cap > 0 else 0,
        })

    @app.get("/api/usage/daily")
    async def api_usage_daily(request: Request) -> JSONResponse:
        """Daily usage breakdown."""
        org = request.state.org
        days = int(request.query_params.get("days", "30"))
        breakdown = billing.get_daily_breakdown(org.id, days=days)
        return JSONResponse(content={
            "org_id": org.id,
            "days": days,
            "daily": [s.model_dump(mode="json") for s in breakdown],
        })

    @app.get("/api/analytics")
    async def api_analytics(request: Request) -> JSONResponse:
        """Reuse existing dashboard analytics."""
        from autorefine.dashboard.api import DashboardAPI

        org = request.state.org
        namespace = org.slug or org.id
        days = int(request.query_params.get("days", "30"))
        api = DashboardAPI(store, prompt_key=namespace)
        return JSONResponse(content=api.get_analytics(days))

    @app.get("/api/prompts")
    async def api_prompts(request: Request) -> JSONResponse:
        """Reuse existing prompt listing."""
        from autorefine.dashboard.api import DashboardAPI

        org = request.state.org
        namespace = org.slug or org.id
        api = DashboardAPI(store, prompt_key=namespace)
        return JSONResponse(content=api.get_prompts())

    return app


def run_cloud_server(
    store: BaseStore,
    authenticator: Authenticator,
    proxy: LLMProxy,
    billing: BillingManager,
    host: str = "0.0.0.0",
    port: int = 8080,
    workers: int = 1,
    cors_origins: str = "*",
) -> None:
    """Start the cloud API server (blocking)."""
    app = create_cloud_app(store, authenticator, proxy, billing, cors_origins)
    if app is None:
        return

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed: pip install autorefine[dashboard]")
        return

    logger.info("Starting AutoRefine Cloud on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, workers=workers, log_level="info")
