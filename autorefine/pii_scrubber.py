"""PII scrubbing — redacts sensitive data before interaction logs reach the refiner model."""

from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger("autorefine.pii_scrubber")

# Pre-compiled patterns for common PII types
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    ("PHONE", re.compile(r"(?<!\d)(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("IP_ADDRESS", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    # US street addresses (basic pattern: number + street name + suffix)
    ("ADDRESS", re.compile(
        r"\b\d{1,6}\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*"
        r"\s+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Ln|Lane|Rd|Road|Ct|Court|Way|Pl|Place)\.?\b",
        re.IGNORECASE,
    )),
    # API keys / tokens (long alphanumeric strings starting with common prefixes)
    ("API_KEY", re.compile(r"\b(?:sk-|pk-|api[_-]?key[_-]?|token[_-]?)[a-zA-Z0-9_-]{20,}\b")),
]


class PIIScrubber:
    """Redacts PII from text before it is sent to the refiner model.

    Usage::

        scrubber = PIIScrubber()
        clean = scrubber.scrub("Email me at alice@example.com")
        # "Email me at [EMAIL_REDACTED]"

    You can add custom patterns or a custom scrub function::

        scrubber = PIIScrubber(custom_patterns=[("MRN", re.compile(r"MRN-\\d{8}"))])
        # or
        scrubber = PIIScrubber(custom_scrub_fn=my_fn)
    """

    def __init__(
        self,
        enabled: bool = True,
        custom_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
        custom_scrub_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._enabled = enabled
        self._patterns = list(_PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)
        self._custom_fn = custom_scrub_fn

    def scrub(self, text: str) -> str:
        """Return *text* with PII patterns replaced by redaction markers."""
        if not self._enabled or not text:
            return text

        result = text
        for label, pattern in self._patterns:
            result = pattern.sub(f"[{label}_REDACTED]", result)

        if self._custom_fn:
            result = self._custom_fn(result)

        return result

    def scrub_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Scrub a list of message dicts (role/content pairs) in place."""
        return [
            {**m, "content": self.scrub(m.get("content", ""))}
            for m in messages
        ]
