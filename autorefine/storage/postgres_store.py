"""PostgreSQL storage backend for multi-server production deployments.

Uses ``asyncpg`` with connection pooling for high-throughput async access.
Every ``BaseStore`` method has a synchronous wrapper that spins up an
event loop internally, so the store works in non-async code too.

Features:
- Automatic table creation on ``initialize()``
- Connection pooling (2–10 connections by default)
- Row-level locking on prompt version activation via ``SELECT … FOR UPDATE``
- JSONB columns for messages / metadata
- Proper indexes on timestamps, interaction IDs, and status columns

Requires: ``pip install autorefine[postgres]``
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from autorefine.models import (
    ABTest,
    CostEntry,
    FeedbackSignal,
    Interaction,
    PromptVersion,
)
from autorefine.storage.base import BaseStore

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError:
    asyncpg = None  # type: ignore[assignment]


# ── Full SQL schema ──────────────────────────────────────────────────

SCHEMA = """\
-- Interaction log.
CREATE TABLE IF NOT EXISTS interactions (
    id              TEXT PRIMARY KEY,
    prompt_key      TEXT             NOT NULL DEFAULT 'default',
    prompt_version  INTEGER                   DEFAULT 0,
    system_prompt   TEXT                      DEFAULT '',
    messages        JSONB                     DEFAULT '[]',
    response_text   TEXT                      DEFAULT '',
    input_tokens    INTEGER                   DEFAULT 0,
    output_tokens   INTEGER                   DEFAULT 0,
    model           TEXT                      DEFAULT '',
    provider        TEXT                      DEFAULT '',
    cost_usd        DOUBLE PRECISION          DEFAULT 0.0,
    created_at      TIMESTAMPTZ      NOT NULL,
    metadata        JSONB                     DEFAULT '{}'
);

-- User feedback.
CREATE TABLE IF NOT EXISTS feedback (
    id              TEXT PRIMARY KEY,
    interaction_id  TEXT    NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    feedback_type   TEXT    NOT NULL,
    score           DOUBLE PRECISION          DEFAULT 0.0,
    confidence      DOUBLE PRECISION          DEFAULT 1.0,
    comment         TEXT                      DEFAULT '',
    correction      TEXT                      DEFAULT '',
    user_id         TEXT                      DEFAULT '',
    processed       BOOLEAN                   DEFAULT FALSE,
    created_at      TIMESTAMPTZ      NOT NULL,
    metadata        JSONB                     DEFAULT '{}'
);

-- Prompt version history.
CREATE TABLE IF NOT EXISTS prompt_versions (
    version           INTEGER NOT NULL,
    prompt_key        TEXT    NOT NULL DEFAULT 'default',
    system_prompt     TEXT    NOT NULL,
    parent_version    INTEGER,
    changelog         TEXT                      DEFAULT '',
    performance_score DOUBLE PRECISION          DEFAULT 0.0,
    interaction_count INTEGER                   DEFAULT 0,
    is_active         BOOLEAN                   DEFAULT TRUE,
    created_at        TIMESTAMPTZ      NOT NULL,
    metadata          JSONB                     DEFAULT '{}',
    PRIMARY KEY (prompt_key, version)
);

-- A/B test state.
CREATE TABLE IF NOT EXISTS ab_tests (
    id                     TEXT PRIMARY KEY,
    prompt_key             TEXT    NOT NULL DEFAULT 'default',
    control_version        INTEGER NOT NULL,
    candidate_version      INTEGER NOT NULL,
    split_ratio            DOUBLE PRECISION DEFAULT 0.2,
    control_interactions   INTEGER          DEFAULT 0,
    candidate_interactions INTEGER          DEFAULT 0,
    control_score          DOUBLE PRECISION DEFAULT 0.0,
    candidate_score        DOUBLE PRECISION DEFAULT 0.0,
    is_active              BOOLEAN          DEFAULT TRUE,
    created_at             TIMESTAMPTZ NOT NULL,
    completed_at           TIMESTAMPTZ,
    result                 TEXT             DEFAULT ''
);

-- Cost tracking.
CREATE TABLE IF NOT EXISTS cost_entries (
    id              TEXT PRIMARY KEY,
    interaction_id  TEXT                      DEFAULT '',
    model           TEXT                      DEFAULT '',
    provider        TEXT                      DEFAULT '',
    input_tokens    INTEGER                   DEFAULT 0,
    output_tokens   INTEGER                   DEFAULT 0,
    cost_usd        DOUBLE PRECISION          DEFAULT 0.0,
    call_type       TEXT                      DEFAULT 'primary',
    created_at      TIMESTAMPTZ      NOT NULL
);

-- Performance indexes.
CREATE INDEX IF NOT EXISTS idx_interactions_key
    ON interactions(prompt_key, created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_created
    ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_interaction
    ON feedback(interaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_processed
    ON feedback(processed, created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_created
    ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_prompt_versions_active
    ON prompt_versions(prompt_key, is_active);
CREATE INDEX IF NOT EXISTS idx_ab_tests_active
    ON ab_tests(prompt_key, is_active);
CREATE INDEX IF NOT EXISTS idx_cost_call_type
    ON cost_entries(call_type, created_at);
CREATE INDEX IF NOT EXISTS idx_cost_created
    ON cost_entries(created_at);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from synchronous code.

    Creates a new event loop if none is running.  This is the mechanism
    that lets every ``BaseStore`` method work without ``await``.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an existing loop (e.g. Jupyter, FastAPI).
        # Fall back to a thread-based approach.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class PostgresStore(BaseStore):
    """Async-capable Postgres backend with connection pooling.

    Args:
        database_url: PostgreSQL connection string
            (e.g. ``"postgresql://user:pass@host:5432/dbname"``).
        min_pool_size: Minimum connections in the pool.
        max_pool_size: Maximum connections in the pool.

    Usage::

        store = PostgresStore("postgresql://user:pass@host/db")
        await store.initialize()   # creates tables, opens pool

        # Sync usage (auto-wraps async internally):
        store.save_interaction(interaction)

        # Async usage:
        await store.async_save_interaction(interaction)

        await store.close()
    """

    def __init__(
        self,
        database_url: str,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        if asyncpg is None:
            raise ImportError(
                "asyncpg is required for PostgresStore. "
                "Install it with: pip install autorefine[postgres]"
            )
        self._dsn = database_url
        self._min_pool = min_pool_size
        self._max_pool = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Create the connection pool and run schema migrations."""
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_pool,
            max_size=self._max_pool,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA)
            await self._migrate(conn)

    async def _migrate(self, conn: Any) -> None:
        """Add columns/tables introduced after the initial schema."""
        # feedback.dimensions and feedback.context
        for col, default in [("dimensions", "'{}'"), ("context", "'{}'" )]:
            try:
                await conn.execute(
                    f"ALTER TABLE feedback ADD COLUMN {col} JSONB DEFAULT {default}"
                )
            except Exception:
                pass  # column already exists
        # refinement_directives table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS refinement_directives (
                prompt_key         TEXT PRIMARY KEY,
                directives         JSONB DEFAULT '[]',
                domain_context     TEXT DEFAULT '',
                preserve_behaviors JSONB DEFAULT '[]',
                version            INTEGER DEFAULT 1,
                created_at         TIMESTAMPTZ NOT NULL,
                updated_at         TIMESTAMPTZ NOT NULL
            )
        """)
        # dimension_schemas table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS dimension_schemas (
                prompt_key  TEXT PRIMARY KEY,
                schema_json JSONB NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL
            )
        """)
        # ab_tests per-dimension columns
        for col in ("control_dimension_scores", "candidate_dimension_scores",
                     "control_dimension_counts", "candidate_dimension_counts"):
            try:
                await conn.execute(
                    f"ALTER TABLE ab_tests ADD COLUMN {col} JSONB DEFAULT '{{}}'"
                )
            except Exception:
                pass

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "PostgresStore not initialised — call "
                "await store.initialize() first"
            )
        return self._pool

    # ══════════════════════════════════════════════════════════════════
    # Synchronous BaseStore wrappers
    # ══════════════════════════════════════════════════════════════════

    def save_interaction(self, interaction: Interaction) -> None:
        _run_sync(self.async_save_interaction(interaction))

    def get_interaction(self, interaction_id: str) -> Interaction | None:
        return _run_sync(self.async_get_interaction(interaction_id))

    def get_interactions(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Interaction]:
        return _run_sync(self.async_get_interactions(prompt_key, limit, since))

    def save_feedback(self, feedback: FeedbackSignal) -> None:
        _run_sync(self.async_save_feedback(feedback))

    def get_feedback(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
        unprocessed_only: bool = False,
    ) -> list[FeedbackSignal]:
        return _run_sync(
            self.async_get_feedback(prompt_key, limit, since, unprocessed_only)
        )

    def mark_feedback_processed(self, feedback_ids: list[str]) -> None:
        _run_sync(self.async_mark_feedback_processed(feedback_ids))

    def save_prompt_version(self, version: PromptVersion) -> None:
        _run_sync(self.async_save_prompt_version(version))

    def get_active_prompt(self, prompt_key: str = "default") -> PromptVersion | None:
        return _run_sync(self.async_get_active_prompt(prompt_key))

    def get_prompt_version(self, prompt_key: str, version: int) -> PromptVersion | None:
        return _run_sync(self.async_get_prompt_version(prompt_key, version))

    def get_prompt_history(self, prompt_key: str = "default") -> list[PromptVersion]:
        return _run_sync(self.async_get_prompt_history(prompt_key))

    def set_active_version(self, prompt_key: str, version: int) -> None:
        _run_sync(self.async_set_active_version(prompt_key, version))

    def save_ab_test(self, ab_test: ABTest) -> None:
        _run_sync(self.async_save_ab_test(ab_test))

    def get_active_ab_test(self, prompt_key: str = "default") -> ABTest | None:
        return _run_sync(self.async_get_active_ab_test(prompt_key))

    def update_ab_test(self, ab_test: ABTest) -> None:
        self.save_ab_test(ab_test)

    def save_cost_entry(self, entry: CostEntry) -> None:
        _run_sync(self.async_save_cost_entry(entry))

    def get_monthly_refiner_cost(self) -> float:
        return _run_sync(self.async_get_monthly_refiner_cost())

    def save_refinement_directives(self, directives: Any) -> None:
        _run_sync(self.async_save_refinement_directives(directives))

    def get_refinement_directives(self, prompt_key: str) -> Any:
        return _run_sync(self.async_get_refinement_directives(prompt_key))

    def save_dimension_schema(self, schema: Any) -> None:
        _run_sync(self.async_save_dimension_schema(schema))

    def get_dimension_schema(self, prompt_key: str) -> Any:
        return _run_sync(self.async_get_dimension_schema(prompt_key))

    def purge_old_data(self, before: datetime) -> int:
        return _run_sync(self.async_purge_old_data(before))

    # ══════════════════════════════════════════════════════════════════
    # Async implementations
    # ══════════════════════════════════════════════════════════════════

    # ── Interactions ─────────────────────────────────────────────────

    async def async_save_interaction(self, interaction: Interaction) -> None:
        pool = self._require_pool()
        data = interaction.model_dump(mode="json")
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO interactions
                   (id, prompt_key, prompt_version, system_prompt, messages,
                    response_text, input_tokens, output_tokens, model,
                    provider, cost_usd, created_at, metadata)
                   VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9,$10,$11,
                           $12::timestamptz,$13::jsonb)
                   ON CONFLICT (id) DO NOTHING""",
                data["id"], data["prompt_key"], data["prompt_version"],
                data["system_prompt"], json.dumps(data["messages"]),
                data["response_text"], data["input_tokens"],
                data["output_tokens"], data["model"], data["provider"],
                data["cost_usd"], data["created_at"],
                json.dumps(data["metadata"]),
            )

    async def async_get_interaction(self, interaction_id: str) -> Interaction | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM interactions WHERE id = $1", interaction_id
            )
        if row is None:
            return None
        return Interaction.model_validate(dict(row))

    async def async_get_interactions(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Interaction]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if since:
                rows = await conn.fetch(
                    """SELECT * FROM interactions
                       WHERE prompt_key=$1 AND created_at >= $2
                       ORDER BY created_at DESC LIMIT $3""",
                    prompt_key, since, limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM interactions
                       WHERE prompt_key=$1
                       ORDER BY created_at DESC LIMIT $2""",
                    prompt_key, limit,
                )
        return [Interaction.model_validate(dict(r)) for r in rows]

    # ── Feedback ─────────────────────────────────────────────────────

    async def async_save_feedback(self, feedback: FeedbackSignal) -> None:
        pool = self._require_pool()
        data = feedback.model_dump(mode="json")
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO feedback
                   (id, interaction_id, feedback_type, score, confidence,
                    comment, correction, user_id, processed, created_at,
                    metadata, dimensions, context)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,FALSE,
                           $9::timestamptz,$10::jsonb,$11::jsonb,$12::jsonb)
                   ON CONFLICT (id) DO NOTHING""",
                data["id"], data["interaction_id"], data["feedback_type"],
                data["score"], data["confidence"], data["comment"],
                data["correction"], data["user_id"], data["created_at"],
                json.dumps(data["metadata"]),
                json.dumps(data.get("dimensions", {})),
                json.dumps(data.get("context", {})),
            )

    async def async_get_feedback(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
        unprocessed_only: bool = False,
    ) -> list[FeedbackSignal]:
        pool = self._require_pool()
        query = """
            SELECT f.* FROM feedback f
            JOIN interactions i ON f.interaction_id = i.id
            WHERE i.prompt_key = $1
        """
        params: list[Any] = [prompt_key]
        idx = 2
        if unprocessed_only:
            query += " AND f.processed = FALSE"
        if since:
            query += f" AND f.created_at >= ${idx}"
            params.append(since)
            idx += 1
        query += f" ORDER BY f.created_at DESC LIMIT ${idx}"
        params.append(limit)
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        results = []
        for r in rows:
            d = dict(r)
            d.pop("processed", None)
            results.append(FeedbackSignal.model_validate(d))
        return results

    async def async_mark_feedback_processed(self, feedback_ids: list[str]) -> None:
        if not feedback_ids:
            return
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE feedback SET processed = TRUE WHERE id = ANY($1::text[])",
                feedback_ids,
            )

    # ── Prompt versions ──────────────────────────────────────────────

    async def async_save_prompt_version(self, version: PromptVersion) -> None:
        pool = self._require_pool()
        data = version.model_dump(mode="json")
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO prompt_versions
                   (version, prompt_key, system_prompt, parent_version,
                    changelog, performance_score, interaction_count,
                    is_active, created_at, metadata)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,
                           $9::timestamptz,$10::jsonb)
                   ON CONFLICT (prompt_key, version) DO UPDATE SET
                    system_prompt   = EXCLUDED.system_prompt,
                    is_active       = EXCLUDED.is_active,
                    performance_score = EXCLUDED.performance_score,
                    interaction_count = EXCLUDED.interaction_count""",
                data["version"], data["prompt_key"],
                data["system_prompt"], data["parent_version"],
                data["changelog"], data["performance_score"],
                data["interaction_count"], data["is_active"],
                data["created_at"], json.dumps(data["metadata"]),
            )

    async def async_get_active_prompt(
        self, prompt_key: str = "default"
    ) -> PromptVersion | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM prompt_versions
                   WHERE prompt_key=$1 AND is_active=TRUE
                   ORDER BY version DESC LIMIT 1""",
                prompt_key,
            )
        if row is None:
            return None
        return PromptVersion.model_validate(dict(row))

    async def async_get_prompt_version(
        self, prompt_key: str, version: int
    ) -> PromptVersion | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM prompt_versions
                   WHERE prompt_key=$1 AND version=$2""",
                prompt_key, version,
            )
        if row is None:
            return None
        return PromptVersion.model_validate(dict(row))

    async def async_get_prompt_history(
        self, prompt_key: str = "default"
    ) -> list[PromptVersion]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM prompt_versions
                   WHERE prompt_key=$1 ORDER BY version""",
                prompt_key,
            )
        return [PromptVersion.model_validate(dict(r)) for r in rows]

    async def async_set_active_version(
        self, prompt_key: str, version: int
    ) -> None:
        """Activate *version* with row-level locking.

        Uses ``SELECT … FOR UPDATE`` inside a transaction to prevent
        concurrent version switches from interleaving.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn, conn.transaction():
            # Lock all rows for this prompt_key
            await conn.fetch(
                """SELECT version FROM prompt_versions
                       WHERE prompt_key=$1 FOR UPDATE""",
                prompt_key,
            )
            await conn.execute(
                """UPDATE prompt_versions
                       SET is_active=FALSE WHERE prompt_key=$1""",
                prompt_key,
            )
            if version >= 0:
                await conn.execute(
                    """UPDATE prompt_versions
                           SET is_active=TRUE
                           WHERE prompt_key=$1 AND version=$2""",
                    prompt_key, version,
                )

    # ── A/B tests ────────────────────────────────────────────────────

    async def async_save_ab_test(self, ab_test: ABTest) -> None:
        pool = self._require_pool()
        data = ab_test.model_dump(mode="json")
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ab_tests
                   (id, prompt_key, control_version, candidate_version,
                    split_ratio, control_interactions,
                    candidate_interactions, control_score,
                    candidate_score, is_active, created_at,
                    completed_at, result)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                           $11::timestamptz,$12,$13)
                   ON CONFLICT (id) DO UPDATE SET
                    control_interactions  = EXCLUDED.control_interactions,
                    candidate_interactions = EXCLUDED.candidate_interactions,
                    control_score         = EXCLUDED.control_score,
                    candidate_score       = EXCLUDED.candidate_score,
                    is_active             = EXCLUDED.is_active,
                    completed_at          = EXCLUDED.completed_at,
                    result                = EXCLUDED.result""",
                data["id"], data["prompt_key"],
                data["control_version"], data["candidate_version"],
                data["split_ratio"], data["control_interactions"],
                data["candidate_interactions"], data["control_score"],
                data["candidate_score"], data["is_active"],
                data["created_at"], data.get("completed_at"),
                data["result"],
            )

    async def async_get_active_ab_test(
        self, prompt_key: str = "default"
    ) -> ABTest | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM ab_tests
                   WHERE prompt_key=$1 AND is_active=TRUE
                   ORDER BY created_at DESC LIMIT 1""",
                prompt_key,
            )
        if row is None:
            return None
        return ABTest.model_validate(dict(row))

    # ── Cost tracking ────────────────────────────────────────────────

    async def async_save_cost_entry(self, entry: CostEntry) -> None:
        pool = self._require_pool()
        data = entry.model_dump(mode="json")
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO cost_entries
                   (id, interaction_id, model, provider, input_tokens,
                    output_tokens, cost_usd, call_type, created_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::timestamptz)
                   ON CONFLICT (id) DO NOTHING""",
                data["id"], data["interaction_id"], data["model"],
                data["provider"], data["input_tokens"],
                data["output_tokens"], data["cost_usd"],
                data["call_type"], data["created_at"],
            )

    async def async_get_monthly_refiner_cost(self) -> float:
        pool = self._require_pool()
        now = _utc_now()
        month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COALESCE(SUM(cost_usd), 0) AS total
                   FROM cost_entries
                   WHERE call_type='refiner' AND created_at >= $1""",
                month_start,
            )
        return float(row["total"]) if row else 0.0

    # ── Refinement directives ────────────────────────────────────────

    async def async_save_refinement_directives(self, directives: Any) -> None:
        pool = self._require_pool()
        data = directives.model_dump(mode="json") if hasattr(directives, "model_dump") else directives
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO refinement_directives
                   (prompt_key, directives, domain_context, preserve_behaviors,
                    version, created_at, updated_at)
                   VALUES ($1, $2::jsonb, $3, $4::jsonb, $5,
                           $6::timestamptz, $7::timestamptz)
                   ON CONFLICT (prompt_key) DO UPDATE SET
                    directives = EXCLUDED.directives,
                    domain_context = EXCLUDED.domain_context,
                    preserve_behaviors = EXCLUDED.preserve_behaviors,
                    version = EXCLUDED.version,
                    updated_at = EXCLUDED.updated_at""",
                data["prompt_key"],
                json.dumps(data.get("directives", [])),
                data.get("domain_context", ""),
                json.dumps(data.get("preserve_behaviors", [])),
                data.get("version", 1),
                data.get("created_at", _utc_now().isoformat()),
                data.get("updated_at", _utc_now().isoformat()),
            )

    async def async_get_refinement_directives(self, prompt_key: str) -> Any:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM refinement_directives WHERE prompt_key = $1",
                prompt_key,
            )
        if row is None:
            return None
        from autorefine.directives import RefinementDirectives
        return RefinementDirectives.model_validate(dict(row))

    # ── Dimension schemas ────────────────────────────────────────────

    async def async_save_dimension_schema(self, schema: Any) -> None:
        pool = self._require_pool()
        data = schema.model_dump(mode="json") if hasattr(schema, "model_dump") else schema
        pk = data.get("prompt_key", "default")
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO dimension_schemas
                   (prompt_key, schema_json, created_at)
                   VALUES ($1, $2::jsonb, $3::timestamptz)
                   ON CONFLICT (prompt_key) DO UPDATE SET
                    schema_json = EXCLUDED.schema_json""",
                pk, json.dumps(data),
                data.get("created_at", _utc_now().isoformat()),
            )

    async def async_get_dimension_schema(self, prompt_key: str) -> Any:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT schema_json FROM dimension_schemas WHERE prompt_key = $1",
                prompt_key,
            )
        if row is None:
            return None
        from autorefine.dimensions import FeedbackDimensionSchema
        raw = row["schema_json"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        return FeedbackDimensionSchema.model_validate(raw)

    # ── Maintenance ──────────────────────────────────────────────────

    async def async_purge_old_data(self, before: datetime) -> int:
        pool = self._require_pool()
        total = 0
        async with pool.acquire() as conn, conn.transaction():
            # Delete in dependency order (feedback references interactions)
            for table in ("feedback", "cost_entries", "interactions"):
                result = await conn.execute(
                    f"DELETE FROM {table} WHERE created_at < $1",
                    before,
                )
                # asyncpg returns "DELETE N"
                total += int(result.split()[-1])
        return total
