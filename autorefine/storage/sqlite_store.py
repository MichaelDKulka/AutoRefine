"""SQLite storage backend for single-server production deployments.

Uses WAL (Write-Ahead Logging) mode for concurrent read/write access,
proper indexes on timestamp and foreign-key columns, and automatic table
creation on first use.  A :class:`threading.Lock` serialises all writes
so the store is safe for multi-threaded applications.

Good for: production single-process / single-server deployments.
Not for: multi-server / horizontally-scaled systems (use Postgres).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from autorefine.models import (
    ABTest,
    CostEntry,
    FeedbackSignal,
    Interaction,
    PromptVersion,
)
from autorefine.storage.base import BaseStore

# ── Full SQL schema ──────────────────────────────────────────────────

SCHEMA = """\
-- Interaction log: every LLM request/response pair.
CREATE TABLE IF NOT EXISTS interactions (
    id              TEXT PRIMARY KEY,
    prompt_key      TEXT    NOT NULL DEFAULT 'default',
    prompt_version  INTEGER          DEFAULT 0,
    system_prompt   TEXT             DEFAULT '',
    messages        TEXT             DEFAULT '[]',   -- JSON array
    response_text   TEXT             DEFAULT '',
    input_tokens    INTEGER          DEFAULT 0,
    output_tokens   INTEGER          DEFAULT 0,
    model           TEXT             DEFAULT '',
    provider        TEXT             DEFAULT '',
    cost_usd        REAL             DEFAULT 0.0,
    created_at      TEXT    NOT NULL,                -- ISO-8601 UTC
    metadata        TEXT             DEFAULT '{}'    -- JSON object
);

-- User feedback on interactions.
CREATE TABLE IF NOT EXISTS feedback (
    id              TEXT PRIMARY KEY,
    interaction_id  TEXT    NOT NULL,
    feedback_type   TEXT    NOT NULL,
    score           REAL             DEFAULT 0.0,
    confidence      REAL             DEFAULT 1.0,
    comment         TEXT             DEFAULT '',
    correction      TEXT             DEFAULT '',
    user_id         TEXT             DEFAULT '',
    processed       INTEGER          DEFAULT 0,      -- 0 = unprocessed
    created_at      TEXT    NOT NULL,
    metadata        TEXT             DEFAULT '{}'
);

-- Immutable prompt version history.
CREATE TABLE IF NOT EXISTS prompt_versions (
    version           INTEGER NOT NULL,
    prompt_key        TEXT    NOT NULL DEFAULT 'default',
    system_prompt     TEXT    NOT NULL,
    parent_version    INTEGER,
    changelog         TEXT             DEFAULT '',
    performance_score REAL             DEFAULT 0.0,
    interaction_count INTEGER          DEFAULT 0,
    is_active         INTEGER          DEFAULT 1,    -- boolean
    created_at        TEXT    NOT NULL,
    metadata          TEXT             DEFAULT '{}',
    PRIMARY KEY (prompt_key, version)
);

-- A/B test state.
CREATE TABLE IF NOT EXISTS ab_tests (
    id                    TEXT PRIMARY KEY,
    prompt_key            TEXT    NOT NULL DEFAULT 'default',
    control_version       INTEGER NOT NULL,
    candidate_version     INTEGER NOT NULL,
    split_ratio           REAL             DEFAULT 0.2,
    control_interactions  INTEGER          DEFAULT 0,
    candidate_interactions INTEGER         DEFAULT 0,
    control_score         REAL             DEFAULT 0.0,
    candidate_score       REAL             DEFAULT 0.0,
    is_active             INTEGER          DEFAULT 1,
    created_at            TEXT    NOT NULL,
    completed_at          TEXT,
    result                TEXT             DEFAULT ''
);

-- Cost tracking for all LLM API calls.
CREATE TABLE IF NOT EXISTS cost_entries (
    id              TEXT PRIMARY KEY,
    interaction_id  TEXT             DEFAULT '',
    model           TEXT             DEFAULT '',
    provider        TEXT             DEFAULT '',
    input_tokens    INTEGER          DEFAULT 0,
    output_tokens   INTEGER          DEFAULT 0,
    cost_usd        REAL             DEFAULT 0.0,
    call_type       TEXT             DEFAULT 'primary',  -- 'primary' | 'refiner'
    created_at      TEXT    NOT NULL
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


class SQLiteStore(BaseStore):
    """Production-ready SQLite backend with WAL mode and automatic schema.

    Args:
        path: Database file path.  Parent directories are created
            automatically.  Defaults to ``~/.autorefine/store.db``.
    """

    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = str(Path.home() / ".autorefine" / "store.db")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(SCHEMA)
        conn.close()

    # ── Interactions ─────────────────────────────────────────────────

    def save_interaction(self, interaction: Interaction) -> None:
        data = interaction.model_dump(mode="json")
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT OR REPLACE INTO interactions
                   (id, prompt_key, prompt_version, system_prompt, messages,
                    response_text, input_tokens, output_tokens, model, provider,
                    cost_usd, created_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    data["id"], data["prompt_key"], data["prompt_version"],
                    data["system_prompt"], json.dumps(data["messages"]),
                    data["response_text"], data["input_tokens"],
                    data["output_tokens"], data["model"], data["provider"],
                    data["cost_usd"], data["created_at"],
                    json.dumps(data["metadata"]),
                ),
            )
            conn.commit()
            conn.close()

    def get_interaction(self, interaction_id: str) -> Interaction | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return self._row_to_interaction(row)

    def get_interactions(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Interaction]:
        conn = self._connect()
        if since:
            rows = conn.execute(
                """SELECT * FROM interactions
                   WHERE prompt_key = ? AND created_at >= ?
                   ORDER BY created_at DESC LIMIT ?""",
                (prompt_key, since.isoformat(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM interactions
                   WHERE prompt_key = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (prompt_key, limit),
            ).fetchall()
        conn.close()
        return [self._row_to_interaction(r) for r in rows]

    @staticmethod
    def _row_to_interaction(row: sqlite3.Row) -> Interaction:
        d = dict(row)
        d["messages"] = json.loads(d.get("messages", "[]"))
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        return Interaction.model_validate(d)

    # ── Feedback ─────────────────────────────────────────────────────

    def save_feedback(self, feedback: FeedbackSignal) -> None:
        data = feedback.model_dump(mode="json")
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT OR REPLACE INTO feedback
                   (id, interaction_id, feedback_type, score, confidence,
                    comment, correction, user_id, processed, created_at,
                    metadata)
                   VALUES (?,?,?,?,?,?,?,?,0,?,?)""",
                (
                    data["id"], data["interaction_id"], data["feedback_type"],
                    data["score"], data["confidence"], data["comment"],
                    data["correction"], data["user_id"], data["created_at"],
                    json.dumps(data["metadata"]),
                ),
            )
            conn.commit()
            conn.close()

    def get_feedback(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
        unprocessed_only: bool = False,
    ) -> list[FeedbackSignal]:
        conn = self._connect()
        query = """
            SELECT f.* FROM feedback f
            JOIN interactions i ON f.interaction_id = i.id
            WHERE i.prompt_key = ?
        """
        params: list = [prompt_key]
        if unprocessed_only:
            query += " AND f.processed = 0"
        if since:
            query += " AND f.created_at >= ?"
            params.append(since.isoformat())
        query += " ORDER BY f.created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [self._row_to_feedback(r) for r in rows]

    def mark_feedback_processed(self, feedback_ids: list[str]) -> None:
        if not feedback_ids:
            return
        with self._lock:
            conn = self._connect()
            placeholders = ",".join("?" for _ in feedback_ids)
            conn.execute(
                f"UPDATE feedback SET processed = 1 WHERE id IN ({placeholders})",
                feedback_ids,
            )
            conn.commit()
            conn.close()

    @staticmethod
    def _row_to_feedback(row: sqlite3.Row) -> FeedbackSignal:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        d.pop("processed", None)
        return FeedbackSignal.model_validate(d)

    # ── Prompt versions ──────────────────────────────────────────────

    def save_prompt_version(self, version: PromptVersion) -> None:
        data = version.model_dump(mode="json")
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT OR REPLACE INTO prompt_versions
                   (version, prompt_key, system_prompt, parent_version,
                    changelog, performance_score, interaction_count,
                    is_active, created_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    data["version"], data["prompt_key"],
                    data["system_prompt"], data["parent_version"],
                    data["changelog"], data["performance_score"],
                    data["interaction_count"], int(data["is_active"]),
                    data["created_at"], json.dumps(data["metadata"]),
                ),
            )
            conn.commit()
            conn.close()

    def get_active_prompt(self, prompt_key: str = "default") -> PromptVersion | None:
        conn = self._connect()
        row = conn.execute(
            """SELECT * FROM prompt_versions
               WHERE prompt_key = ? AND is_active = 1
               ORDER BY version DESC LIMIT 1""",
            (prompt_key,),
        ).fetchone()
        conn.close()
        return self._row_to_prompt_version(row) if row else None

    def get_prompt_version(self, prompt_key: str, version: int) -> PromptVersion | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM prompt_versions WHERE prompt_key = ? AND version = ?",
            (prompt_key, version),
        ).fetchone()
        conn.close()
        return self._row_to_prompt_version(row) if row else None

    def get_prompt_history(self, prompt_key: str = "default") -> list[PromptVersion]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM prompt_versions WHERE prompt_key = ? ORDER BY version",
            (prompt_key,),
        ).fetchall()
        conn.close()
        return [self._row_to_prompt_version(r) for r in rows]

    def set_active_version(self, prompt_key: str, version: int) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "UPDATE prompt_versions SET is_active = 0 WHERE prompt_key = ?",
                (prompt_key,),
            )
            if version >= 0:
                conn.execute(
                    "UPDATE prompt_versions SET is_active = 1 WHERE prompt_key = ? AND version = ?",
                    (prompt_key, version),
                )
            conn.commit()
            conn.close()

    @staticmethod
    def _row_to_prompt_version(row: sqlite3.Row) -> PromptVersion:
        d = dict(row)
        d["is_active"] = bool(d.get("is_active"))
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        return PromptVersion.model_validate(d)

    # ── A/B tests ────────────────────────────────────────────────────

    def save_ab_test(self, ab_test: ABTest) -> None:
        data = ab_test.model_dump(mode="json")
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT OR REPLACE INTO ab_tests
                   (id, prompt_key, control_version, candidate_version,
                    split_ratio, control_interactions,
                    candidate_interactions, control_score,
                    candidate_score, is_active, created_at,
                    completed_at, result)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    data["id"], data["prompt_key"],
                    data["control_version"], data["candidate_version"],
                    data["split_ratio"], data["control_interactions"],
                    data["candidate_interactions"], data["control_score"],
                    data["candidate_score"], int(data["is_active"]),
                    data["created_at"], data.get("completed_at"),
                    data["result"],
                ),
            )
            conn.commit()
            conn.close()

    def get_active_ab_test(self, prompt_key: str = "default") -> ABTest | None:
        conn = self._connect()
        row = conn.execute(
            """SELECT * FROM ab_tests
               WHERE prompt_key = ? AND is_active = 1
               ORDER BY created_at DESC LIMIT 1""",
            (prompt_key,),
        ).fetchone()
        conn.close()
        return self._row_to_ab_test(row) if row else None

    def update_ab_test(self, ab_test: ABTest) -> None:
        self.save_ab_test(ab_test)

    @staticmethod
    def _row_to_ab_test(row: sqlite3.Row) -> ABTest:
        d = dict(row)
        d["is_active"] = bool(d.get("is_active"))
        return ABTest.model_validate(d)

    # ── Cost tracking ────────────────────────────────────────────────

    def save_cost_entry(self, entry: CostEntry) -> None:
        data = entry.model_dump(mode="json")
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT OR REPLACE INTO cost_entries
                   (id, interaction_id, model, provider, input_tokens,
                    output_tokens, cost_usd, call_type, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    data["id"], data["interaction_id"], data["model"],
                    data["provider"], data["input_tokens"],
                    data["output_tokens"], data["cost_usd"],
                    data["call_type"], data["created_at"],
                ),
            )
            conn.commit()
            conn.close()

    def get_monthly_refiner_cost(self) -> float:
        now = _utc_now()
        month_start = f"{now.strftime('%Y-%m')}-01T00:00:00"
        conn = self._connect()
        row = conn.execute(
            """SELECT COALESCE(SUM(cost_usd), 0) AS total
               FROM cost_entries
               WHERE call_type = 'refiner' AND created_at >= ?""",
            (month_start,),
        ).fetchone()
        conn.close()
        return float(row["total"]) if row else 0.0

    # ── Maintenance ──────────────────────────────────────────────────

    def purge_old_data(self, before: datetime) -> int:
        cutoff = before.isoformat()
        total = 0
        with self._lock:
            conn = self._connect()
            for table in ("feedback", "cost_entries", "interactions"):
                cursor = conn.execute(
                    f"DELETE FROM {table} WHERE created_at < ?", (cutoff,)
                )
                total += cursor.rowcount
            conn.commit()
            conn.close()
        return total
