"""FastAPI dashboard server with session auth, CORS config, and rate limiting.

Serves the single-page dashboard UI and the REST API.
Falls back gracefully if FastAPI/uvicorn are not installed.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from autorefine.dashboard.api import DashboardAPI
from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.dashboard")
TEMPLATE_DIR = Path(__file__).parent / "templates"


# ── Rate limiter ─────────────────────────────────────────────────────

class _RateLimiter:
    """Simple in-memory sliding window rate limiter."""

    def __init__(self, max_per_second: int = 10):
        self._max = max_per_second
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, client_ip: str) -> bool:
        now = time.monotonic()
        window = self._hits[client_ip]
        # Prune old entries
        self._hits[client_ip] = [t for t in window if now - t < 1.0]
        if len(self._hits[client_ip]) >= self._max:
            return False
        self._hits[client_ip].append(now)
        return True


def _hash_password(password: str) -> str:
    """Hash a password with SHA-256 + salt for session token generation."""
    return hashlib.sha256(password.encode()).hexdigest()


def create_app(
    store: BaseStore,
    prompt_key: str = "default",
    password: str = "",
    cors_origins: str = "",
    rate_limit: int = 10,
) -> Any:
    """Create the FastAPI application with security hardening.

    Returns ``None`` if FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, Query, Request
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        logger.warning(
            "FastAPI not installed -- dashboard unavailable. "
            "pip install autorefine[dashboard]"
        )
        return None

    api = DashboardAPI(store, prompt_key)
    app = FastAPI(title="AutoRefine Dashboard", docs_url=None, redoc_url=None)
    limiter = _RateLimiter(max_per_second=rate_limit)

    # ── Session-based auth ───────────────────────────────────────────

    _sessions: set[str] = set()
    _password_hash = _hash_password(password) if password else ""

    if password:
        @app.post("/api/auth/login")
        async def login(request: Request):
            data = await request.json()
            pwd = data.get("password", "")
            if secrets.compare_digest(_hash_password(pwd), _password_hash):
                token = secrets.token_urlsafe(32)
                _sessions.add(token)
                return {"status": "ok", "token": token}
            return JSONResponse(status_code=401, content={"error": "Invalid password"})

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            path = request.url.path
            # Allow login, widget feedback, and OPTIONS without auth
            if path in ("/api/auth/login", "/api/widget/feedback") or request.method == "OPTIONS":
                return await call_next(request)
            if path.startswith("/api") or path == "/":
                token = request.headers.get("Authorization", "").replace("Bearer ", "")
                # Also accept basic auth for backward compat
                auth_header = request.headers.get("Authorization", "")
                if token in _sessions:
                    return await call_next(request)
                if auth_header.startswith("Basic "):
                    import base64
                    try:
                        decoded = base64.b64decode(auth_header[6:]).decode()
                        _, pwd = decoded.split(":", 1)
                        if secrets.compare_digest(_hash_password(pwd), _password_hash):
                            return await call_next(request)
                    except Exception:
                        pass
                return JSONResponse(
                    status_code=401,
                    content={"error": "Unauthorized"},
                    headers={"WWW-Authenticate": 'Bearer realm="AutoRefine"'},
                )
            return await call_next(request)

    # ── Rate limiting middleware ──────────────────────────────────────

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/api"):
            client_ip = request.client.host if request.client else "unknown"
            if not limiter.allow(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"},
                )
        return await call_next(request)

    # ── CORS middleware (configurable origins) ────────────────────────

    allowed_origins = [o.strip() for o in cors_origins.split(",") if o.strip()] if cors_origins else []

    @app.middleware("http")
    async def cors_middleware(request: Request, call_next):
        origin = request.headers.get("Origin", "")
        if request.method == "OPTIONS":
            headers = {
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            }
            if "*" in allowed_origins or origin in allowed_origins:
                headers["Access-Control-Allow-Origin"] = origin or "*"
            elif not allowed_origins:
                # No CORS config = same-origin only (don't set the header)
                pass
            return JSONResponse(content={}, headers=headers)
        response = await call_next(request)
        if origin and ("*" in allowed_origins or origin in allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
        return response

    # ── Dashboard UI ─────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        template = TEMPLATE_DIR / "index.html"
        if template.exists():
            return template.read_text(encoding="utf-8")
        return "<h1>AutoRefine Dashboard</h1><p>Template not found.</p>"

    # ── API routes ───────────────────────────────────────────────────

    @app.get("/api/prompts")
    async def prompts():
        return api.get_prompts()

    @app.get("/api/prompts/{key}/history")
    async def prompt_history(key: str):
        return api.get_prompt_history(key)

    @app.post("/api/prompts/{key}/rollback/{version}")
    async def rollback(key: str, version: int):
        return api.rollback(version, key)

    @app.get("/api/analytics")
    async def analytics(days: int = Query(30, ge=1, le=365), prompt_key: str = Query("")):
        return api.get_analytics(days, prompt_key)

    @app.get("/api/feedback")
    async def feedback(limit: int = Query(50, ge=1, le=500), prompt_key: str = Query(""),
                       signal_type: str = Query(""), since_days: int = Query(0, ge=0)):
        return api.get_feedback(limit, prompt_key, signal_type, since_days)

    @app.get("/api/ab-tests")
    async def ab_tests(prompt_key: str = Query("")):
        return api.get_ab_tests(prompt_key)

    @app.post("/api/ab-tests/{key}/promote")
    async def promote(key: str, test_id: str = Query("")):
        return api.promote_ab_test(test_id)

    @app.post("/api/ab-tests/{key}/reject")
    async def reject(key: str, test_id: str = Query("")):
        return api.reject_ab_test(test_id)

    @app.get("/api/costs")
    async def costs():
        return api.get_costs()

    @app.get("/api/usage")
    async def usage():
        return api.get_usage_summary()

    @app.get("/api/usage/daily")
    async def usage_daily(days: int = Query(30, ge=1, le=365)):
        return {"daily": api.get_daily_usage(days=days)}

    # ── Widget feedback endpoint (always public) ─────────────────────

    from autorefine.dashboard.widget_endpoint import WidgetFeedbackHandler
    widget_handler = WidgetFeedbackHandler(store, prompt_key)

    @app.post("/api/widget/feedback")
    async def widget_feedback(request: Request):
        data = await request.json()
        result = widget_handler.handle(data)
        status_code = 200 if result.get("status") == "ok" else 400
        return JSONResponse(content=result, status_code=status_code)

    return app


def run_dashboard(
    store: BaseStore,
    prompt_key: str = "default",
    port: int = 8787,
    password: str = "",
    cors_origins: str = "",
    rate_limit: int = 10,
) -> None:
    """Start the dashboard server (blocking)."""
    app = create_app(store, prompt_key, password, cors_origins, rate_limit)
    if app is None:
        logger.error("Cannot start dashboard -- install FastAPI: pip install autorefine[dashboard]")
        return

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed: pip install autorefine[dashboard]")
        return

    logger.info("Starting AutoRefine dashboard on http://localhost:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
