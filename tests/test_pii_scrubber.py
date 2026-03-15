"""Tests for PII scrubbing."""

from __future__ import annotations

import re

from autorefine.pii_scrubber import PIIScrubber


class TestPIIScrubber:
    def test_scrubs_email(self):
        scrubber = PIIScrubber()
        assert scrubber.scrub("Contact alice@example.com") == "Contact [EMAIL_REDACTED]"

    def test_scrubs_phone(self):
        scrubber = PIIScrubber()
        result = scrubber.scrub("Call me at 555-123-4567")
        assert "[PHONE_REDACTED]" in result
        assert "555-123-4567" not in result

    def test_scrubs_ssn(self):
        scrubber = PIIScrubber()
        result = scrubber.scrub("My SSN is 123-45-6789")
        assert "[SSN_REDACTED]" in result
        assert "123-45-6789" not in result

    def test_scrubs_api_key(self):
        scrubber = PIIScrubber()
        result = scrubber.scrub("Use key sk-abc123def456ghi789jkl012mno")
        assert "[API_KEY_REDACTED]" in result
        assert "sk-abc123" not in result

    def test_scrubs_ip_address(self):
        scrubber = PIIScrubber()
        result = scrubber.scrub("Server at 192.168.1.100")
        assert "[IP_ADDRESS_REDACTED]" in result

    def test_disabled_returns_original(self):
        scrubber = PIIScrubber(enabled=False)
        text = "Email alice@example.com"
        assert scrubber.scrub(text) == text

    def test_empty_string(self):
        scrubber = PIIScrubber()
        assert scrubber.scrub("") == ""

    def test_no_pii_passes_through(self):
        scrubber = PIIScrubber()
        text = "How do I make pasta?"
        assert scrubber.scrub(text) == text

    def test_custom_pattern(self):
        custom = [("MRN", re.compile(r"MRN-\d{8}"))]
        scrubber = PIIScrubber(custom_patterns=custom)
        result = scrubber.scrub("Patient MRN-12345678")
        assert "[MRN_REDACTED]" in result

    def test_custom_scrub_fn(self):
        def my_fn(text: str) -> str:
            return text.replace("SECRET", "[REDACTED]")

        scrubber = PIIScrubber(custom_scrub_fn=my_fn)
        assert scrubber.scrub("The SECRET code") == "The [REDACTED] code"

    def test_scrub_messages(self):
        scrubber = PIIScrubber()
        messages = [
            {"role": "user", "content": "My email is alice@example.com"},
            {"role": "assistant", "content": "Got it!"},
        ]
        result = scrubber.scrub_messages(messages)
        assert "[EMAIL_REDACTED]" in result[0]["content"]
        assert result[1]["content"] == "Got it!"

    def test_multiple_pii_in_one_string(self):
        scrubber = PIIScrubber()
        text = "Email alice@example.com and call 555-123-4567"
        result = scrubber.scrub(text)
        assert "[EMAIL_REDACTED]" in result
        assert "[PHONE_REDACTED]" in result
