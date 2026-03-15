"""Privacy utilities — encryption at rest, API key scrubbing, enhanced PII redaction.

This module provides:
- Fernet encryption/decryption for data at rest (optional dependency: cryptography)
- API key scrubbing from Interaction objects before storage
- Namespace-aware prompt key prefixing for multi-tenant isolation
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("autorefine.privacy")

# ── API key patterns to scrub before storage ─────────────────────────

_KEY_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),            # OpenAI
    re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}"),         # Anthropic
    re.compile(r"pk-[a-zA-Z0-9_-]{20,}"),             # generic
    re.compile(r"api[_-]?key[_-]?[a-zA-Z0-9_-]{16,}"),  # generic
    re.compile(r"bearer\s+[a-zA-Z0-9_.-]{20,}", re.IGNORECASE),
    re.compile(r"token[_-]?[a-zA-Z0-9_-]{20,}"),
]


def scrub_api_keys(text: str) -> str:
    """Replace any API keys found in text with [KEY_REDACTED]."""
    if not text:
        return text
    result = text
    for pattern in _KEY_PATTERNS:
        result = pattern.sub("[KEY_REDACTED]", result)
    return result


def scrub_interaction_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Scrub API keys from an Interaction dict before storage.

    Processes: system_prompt, response_text, messages, metadata.
    """
    if "system_prompt" in data and isinstance(data["system_prompt"], str):
        data["system_prompt"] = scrub_api_keys(data["system_prompt"])
    if "response_text" in data and isinstance(data["response_text"], str):
        data["response_text"] = scrub_api_keys(data["response_text"])
    if "messages" in data and isinstance(data["messages"], list):
        for msg in data["messages"]:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = scrub_api_keys(msg["content"])
    if "metadata" in data and isinstance(data["metadata"], dict):
        for k, v in data["metadata"].items():
            if isinstance(v, str):
                data["metadata"][k] = scrub_api_keys(v)
    return data


# ── Namespace prefixing ──────────────────────────────────────────────

def namespace_key(namespace: str, prompt_key: str) -> str:
    """Prefix a prompt_key with the tenant namespace."""
    if not namespace:
        return prompt_key
    return f"{namespace}:{prompt_key}"


# ── Encryption at rest (optional: requires cryptography) ─────────────

class FieldEncryptor:
    """Encrypts/decrypts string fields using Fernet symmetric encryption.

    Only instantiated when ``encryption_key`` is set in config.
    Requires the ``cryptography`` package.

    Usage::

        enc = FieldEncryptor("your-fernet-key-here")
        cipher = enc.encrypt("sensitive data")
        plain = enc.decrypt(cipher)
    """

    def __init__(self, key: str) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise ImportError(
                "cryptography package required for encryption at rest: "
                "pip install cryptography"
            ) from exc
        self._fernet = Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string, returning a base64-encoded ciphertext string."""
        if not plaintext:
            return plaintext
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded ciphertext string."""
        if not ciphertext:
            return ciphertext
        try:
            return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
        except Exception:
            # If decryption fails (wrong key, not encrypted), return as-is
            logger.debug("Decryption failed — returning raw value")
            return ciphertext

    def encrypt_dict_fields(self, data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        """Encrypt specific string fields in a dict."""
        for field in fields:
            if field in data and isinstance(data[field], str) and data[field]:
                data[field] = self.encrypt(data[field])
        return data

    def decrypt_dict_fields(self, data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        """Decrypt specific string fields in a dict."""
        for field in fields:
            if field in data and isinstance(data[field], str) and data[field]:
                data[field] = self.decrypt(data[field])
        return data
