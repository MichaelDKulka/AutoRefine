"""Refinement directives — developer-authored constraints for the refiner.

Directives are explicit instructions that represent domain knowledge,
hard constraints, and strategic priorities the refiner cannot infer
from feedback alone.  They persist in the store alongside the prompt_key.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from autorefine.storage.base import BaseStore

logger = logging.getLogger("autorefine.directives")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RefinementDirectives(BaseModel):
    """Persisted directive set for a prompt_key."""

    prompt_key: str = "default"
    directives: list[str] = Field(default_factory=list)
    domain_context: str = ""
    preserve_behaviors: list[str] = Field(default_factory=list)
    version: int = 1
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class DirectiveManager:
    """Manages directive CRUD and injection into the refinement pipeline."""

    def __init__(self, store: BaseStore) -> None:
        self._store = store

    def set(
        self,
        prompt_key: str,
        directives: list[str] | None = None,
        domain_context: str | None = None,
        preserve_behaviors: list[str] | None = None,
    ) -> RefinementDirectives:
        """Replace all directives for a prompt_key."""
        existing = self._store.get_refinement_directives(prompt_key)
        next_version = (existing.version + 1) if existing else 1
        now = _utc_now()

        rd = RefinementDirectives(
            prompt_key=prompt_key,
            directives=directives if directives is not None else [],
            domain_context=domain_context if domain_context is not None else "",
            preserve_behaviors=preserve_behaviors if preserve_behaviors is not None else [],
            version=next_version,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._store.save_refinement_directives(rd)
        logger.info("Set directives v%d for prompt_key=%s (%d directives)",
                     next_version, prompt_key, len(rd.directives))
        return rd

    def update(
        self,
        prompt_key: str,
        add_directives: list[str] | None = None,
        remove_directives: list[str] | None = None,
        domain_context: str | None = None,
        preserve_behaviors: list[str] | None = None,
    ) -> RefinementDirectives:
        """Merge updates into existing directives."""
        existing = self._store.get_refinement_directives(prompt_key)
        if existing is None:
            existing = RefinementDirectives(prompt_key=prompt_key)

        current = list(existing.directives)
        if remove_directives:
            current = [d for d in current if d not in remove_directives]
        if add_directives:
            current.extend(add_directives)

        return self.set(
            prompt_key=prompt_key,
            directives=current,
            domain_context=domain_context if domain_context is not None else existing.domain_context,
            preserve_behaviors=preserve_behaviors if preserve_behaviors is not None else existing.preserve_behaviors,
        )

    def get(self, prompt_key: str) -> RefinementDirectives | None:
        """Retrieve directives for a prompt_key."""
        return self._store.get_refinement_directives(prompt_key)

    def format_for_meta_prompt(self, prompt_key: str) -> str:
        """Render directives as a formatted block for injection into META_PROMPT.

        Returns an empty string if no directives exist.
        """
        rd = self.get(prompt_key)
        if rd is None:
            return ""
        if not rd.directives and not rd.domain_context and not rd.preserve_behaviors:
            return ""

        lines: list[str] = []
        lines.append("=" * 50)
        lines.append("DEVELOPER DIRECTIVES -- THESE ARE NON-NEGOTIABLE")
        lines.append("=" * 50)

        if rd.domain_context:
            lines.append("")
            lines.append("DOMAIN CONTEXT:")
            lines.append(rd.domain_context)

        if rd.directives:
            lines.append("")
            lines.append("HARD CONSTRAINTS (you MUST respect ALL of these in your revision):")
            for i, d in enumerate(rd.directives, 1):
                lines.append(f"  {i}. {d}")

        if rd.preserve_behaviors:
            lines.append("")
            lines.append("BEHAVIORS TO PRESERVE (do NOT modify or weaken these):")
            for b in rd.preserve_behaviors:
                lines.append(f"  * {b}")

        lines.append("")
        return "\n".join(lines)
