"""Zero-config JSON file storage backend.

Data lives in a single JSON file with an in-memory write-through cache.
Every mutation acquires a :class:`threading.Lock`, serialises the full
in-memory dict to a temporary file, then atomically replaces the target
file via ``os.replace``.  This guarantees that a crash mid-write never
corrupts the store.

Good for: development, single-process CLIs, quick prototypes.
Not for: multi-process or high-throughput production workloads (use
SQLite or Postgres instead).
"""

from __future__ import annotations

import contextlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autorefine.models import (
    ABTest,
    CostEntry,
    FeedbackSignal,
    Interaction,
    PromptVersion,
)
from autorefine.storage.base import BaseStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_EMPTY_STORE: dict[str, Any] = {
    "interactions": [],
    "feedback": [],
    "prompt_versions": [],
    "ab_tests": [],
    "cost_entries": [],
    "processed_feedback_ids": [],
    "refinement_directives": {},
    "dimension_schemas": {},
}


class JSONStore(BaseStore):
    """Single-file JSON store with in-memory cache and atomic writes.

    Args:
        path: File path for the JSON store.  Parent directories are
            created automatically.  Defaults to
            ``~/.autorefine/store.json``.
    """

    def __init__(self, path: str | None = None, namespace: str = "",
                 encryption_key: str = "") -> None:
        if path is None:
            path = str(Path.home() / ".autorefine" / "store.json")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._namespace = namespace
        self._encryptor = None
        if encryption_key:
            from autorefine.privacy import FieldEncryptor
            self._encryptor = FieldEncryptor(encryption_key)
        self._data: dict[str, Any] = self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        """Read the JSON file into memory, or return an empty skeleton."""
        if self._path.exists():
            try:
                raw = self._path.read_text(encoding="utf-8")
                data = json.loads(raw)
                # Ensure all expected keys exist (forward-compat)
                for key, default in _EMPTY_STORE.items():
                    if key not in data:
                        if isinstance(default, list):
                            data[key] = list(default)
                        elif isinstance(default, dict):
                            data[key] = dict(default)
                        else:
                            data[key] = default
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {k: list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v for k, v in _EMPTY_STORE.items()}

    def _flush(self) -> None:
        """Atomically write the in-memory cache to disk.

        Writes to a ``.tmp`` sibling file, then replaces the target via
        ``os.replace`` (atomic on POSIX, near-atomic on Windows).
        """
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(self._data, default=str, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise

    # ── Interactions ─────────────────────────────────────────────────

    def _ns_key(self, prompt_key: str) -> str:
        """Apply namespace prefix to a prompt_key."""
        if self._namespace:
            return f"{self._namespace}:{prompt_key}"
        return prompt_key

    def _encrypt_sensitive(self, data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        if self._encryptor:
            return self._encryptor.encrypt_dict_fields(data, fields)
        return data

    def _decrypt_sensitive(self, data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        if self._encryptor:
            return self._encryptor.decrypt_dict_fields(data, fields)
        return data

    def save_interaction(self, interaction: Interaction) -> None:
        from autorefine.privacy import scrub_interaction_keys
        data = interaction.model_dump(mode="json")
        data = scrub_interaction_keys(data)
        data["prompt_key"] = self._ns_key(data.get("prompt_key", "default"))
        data = self._encrypt_sensitive(data, ["system_prompt", "response_text"])
        with self._lock:
            self._data["interactions"].append(data)
            self._flush()

    def get_interaction(self, interaction_id: str) -> Interaction | None:
        for raw in self._data["interactions"]:
            if raw["id"] == interaction_id:
                decrypted = self._decrypt_sensitive(dict(raw), ["system_prompt", "response_text"])
                return Interaction.model_validate(decrypted)
        return None

    def get_interactions(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Interaction]:
        ns_key = self._ns_key(prompt_key)
        since_iso = since.isoformat() if since else None
        results: list[Interaction] = []
        for raw in reversed(self._data["interactions"]):
            if raw.get("prompt_key") != ns_key:
                continue
            if since_iso and raw.get("created_at", "") < since_iso:
                continue
            decrypted = self._decrypt_sensitive(dict(raw), ["system_prompt", "response_text"])
            results.append(Interaction.model_validate(decrypted))
            if len(results) >= limit:
                break
        return results

    # ── Feedback ─────────────────────────────────────────────────────

    def save_feedback(self, feedback: FeedbackSignal) -> None:
        data = feedback.model_dump(mode="json")
        data = self._encrypt_sensitive(data, ["comment", "correction"])
        with self._lock:
            self._data["feedback"].append(data)
            self._flush()

    def get_feedback(
        self,
        prompt_key: str = "default",
        limit: int = 100,
        since: datetime | None = None,
        unprocessed_only: bool = False,
    ) -> list[FeedbackSignal]:
        ns_key = self._ns_key(prompt_key)
        processed = set(self._data.get("processed_feedback_ids", []))

        ix_keys: dict[str, str] = {}
        for raw in self._data["interactions"]:
            ix_keys[raw["id"]] = raw.get("prompt_key", "default")

        since_iso = since.isoformat() if since else None
        results: list[FeedbackSignal] = []
        for raw in reversed(self._data["feedback"]):
            fid = raw["id"]
            iid = raw.get("interaction_id", "")
            if ix_keys.get(iid, ns_key) != ns_key:
                continue
            if unprocessed_only and fid in processed:
                continue
            if since_iso and raw.get("created_at", "") < since_iso:
                continue
            decrypted = self._decrypt_sensitive(dict(raw), ["comment", "correction"])
            results.append(FeedbackSignal.model_validate(decrypted))
            if len(results) >= limit:
                break
        return results

    def mark_feedback_processed(self, feedback_ids: list[str]) -> None:
        with self._lock:
            existing = set(self._data.get("processed_feedback_ids", []))
            existing.update(feedback_ids)
            self._data["processed_feedback_ids"] = list(existing)
            self._flush()

    # ── Prompt versions ──────────────────────────────────────────────

    def save_prompt_version(self, version: PromptVersion) -> None:
        data = version.model_dump(mode="json")
        data["prompt_key"] = self._ns_key(data.get("prompt_key", "default"))
        data = self._encrypt_sensitive(data, ["system_prompt"])
        with self._lock:
            self._data["prompt_versions"].append(data)
            self._flush()

    def get_active_prompt(self, prompt_key: str = "default") -> PromptVersion | None:
        ns_key = self._ns_key(prompt_key)
        for raw in reversed(self._data["prompt_versions"]):
            if raw.get("prompt_key") == ns_key and raw.get("is_active"):
                decrypted = self._decrypt_sensitive(dict(raw), ["system_prompt"])
                return PromptVersion.model_validate(decrypted)
        return None

    def get_prompt_version(self, prompt_key: str, version: int) -> PromptVersion | None:
        ns_key = self._ns_key(prompt_key)
        for raw in self._data["prompt_versions"]:
            if raw.get("prompt_key") == ns_key and raw.get("version") == version:
                decrypted = self._decrypt_sensitive(dict(raw), ["system_prompt"])
                return PromptVersion.model_validate(decrypted)
        return None

    def get_prompt_history(self, prompt_key: str = "default") -> list[PromptVersion]:
        ns_key = self._ns_key(prompt_key)
        return [
            PromptVersion.model_validate(self._decrypt_sensitive(dict(raw), ["system_prompt"]))
            for raw in self._data["prompt_versions"]
            if raw.get("prompt_key") == ns_key
        ]

    def set_active_version(self, prompt_key: str, version: int) -> None:
        ns_key = self._ns_key(prompt_key)
        with self._lock:
            for raw in self._data["prompt_versions"]:
                if raw.get("prompt_key") == ns_key:
                    raw["is_active"] = (raw["version"] == version)
            self._flush()

    # ── A/B tests ────────────────────────────────────────────────────

    def save_ab_test(self, ab_test: ABTest) -> None:
        with self._lock:
            self._data["ab_tests"].append(ab_test.model_dump(mode="json"))
            self._flush()

    def get_active_ab_test(self, prompt_key: str = "default") -> ABTest | None:
        for raw in reversed(self._data["ab_tests"]):
            if raw.get("prompt_key") == prompt_key and raw.get("is_active"):
                return ABTest.model_validate(raw)
        return None

    def update_ab_test(self, ab_test: ABTest) -> None:
        with self._lock:
            for i, raw in enumerate(self._data["ab_tests"]):
                if raw["id"] == ab_test.id:
                    self._data["ab_tests"][i] = ab_test.model_dump(mode="json")
                    break
            self._flush()

    # ── Cost tracking ────────────────────────────────────────────────

    def save_cost_entry(self, entry: CostEntry) -> None:
        with self._lock:
            self._data["cost_entries"].append(entry.model_dump(mode="json"))
            self._flush()

    def get_monthly_refiner_cost(self) -> float:
        now = _utc_now()
        month_prefix = now.strftime("%Y-%m")
        total = 0.0
        for raw in self._data.get("cost_entries", []):
            if raw.get("call_type") != "refiner":
                continue
            created = raw.get("created_at", "")
            if isinstance(created, str) and created[:7] == month_prefix:
                total += raw.get("cost_usd", 0.0)
        return total

    # ── Refinement directives ────────────────────────────────────────

    def save_refinement_directives(self, directives: Any) -> None:
        data = directives.model_dump(mode="json") if hasattr(directives, "model_dump") else directives
        pk = data.get("prompt_key", "default")
        with self._lock:
            self._data.setdefault("refinement_directives", {})[pk] = data
            self._flush()

    def get_refinement_directives(self, prompt_key: str) -> Any:
        raw = self._data.get("refinement_directives", {}).get(prompt_key)
        if raw is None:
            return None
        from autorefine.directives import RefinementDirectives
        return RefinementDirectives.model_validate(raw)

    # ── Dimension schemas ────────────────────────────────────────────

    def save_dimension_schema(self, schema: Any) -> None:
        data = schema.model_dump(mode="json") if hasattr(schema, "model_dump") else schema
        pk = data.get("prompt_key", "default")
        with self._lock:
            self._data.setdefault("dimension_schemas", {})[pk] = data
            self._flush()

    def get_dimension_schema(self, prompt_key: str) -> Any:
        raw = self._data.get("dimension_schemas", {}).get(prompt_key)
        if raw is None:
            return None
        from autorefine.dimensions import FeedbackDimensionSchema
        return FeedbackDimensionSchema.model_validate(raw)

    # ── Maintenance ──────────────────────────────────────────────────

    def purge_old_data(self, before: datetime) -> int:
        cutoff = before.isoformat()
        count = 0
        with self._lock:
            for key in ("interactions", "feedback", "cost_entries"):
                original = self._data.get(key, [])
                filtered = [r for r in original if r.get("created_at", "") >= cutoff]
                count += len(original) - len(filtered)
                self._data[key] = filtered
            self._flush()
        return count
