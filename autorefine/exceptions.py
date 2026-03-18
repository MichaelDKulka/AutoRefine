"""Custom exception hierarchy for AutoRefine."""

from __future__ import annotations


class AutoRefineError(Exception):
    """Base exception for all AutoRefine errors."""

    def __init__(self, message: str, **context: object) -> None:
        self.context = context
        super().__init__(message)


class ProviderError(AutoRefineError):
    """LLM API call failed (base class for specific provider failures)."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        **context: object,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(
            message, provider=provider, status_code=status_code,
            retryable=retryable, **context,
        )


class ProviderAuthError(ProviderError):
    """API key is invalid or expired (HTTP 401/403)."""

    def __init__(self, message: str, provider: str | None = None, **ctx: object) -> None:
        super().__init__(message, provider=provider, status_code=401, retryable=False, **ctx)


class ProviderRateLimitError(ProviderError):
    """Rate limit hit (HTTP 429). Retryable."""

    def __init__(self, message: str, provider: str | None = None, **ctx: object) -> None:
        super().__init__(message, provider=provider, status_code=429, retryable=True, **ctx)


class ProviderNetworkError(ProviderError):
    """Network-level failure (timeout, DNS, connection refused). Retryable."""

    def __init__(self, message: str, provider: str | None = None, **ctx: object) -> None:
        super().__init__(message, provider=provider, retryable=True, **ctx)


class ProviderResponseError(ProviderError):
    """Provider returned a malformed or unparseable response."""

    def __init__(self, message: str, provider: str | None = None, **ctx: object) -> None:
        super().__init__(message, provider=provider, retryable=False, **ctx)


class RefinementError(AutoRefineError):
    """Refiner model returned invalid or unusable output."""

    def __init__(
        self, message: str, interaction_id: str | None = None, **context: object,
    ) -> None:
        self.interaction_id = interaction_id
        super().__init__(message, interaction_id=interaction_id, **context)


class StorageError(AutoRefineError):
    """Storage backend read/write failed."""

    def __init__(
        self, message: str, backend: str | None = None, **context: object,
    ) -> None:
        self.backend = backend
        super().__init__(message, backend=backend, **context)


class CostLimitExceeded(AutoRefineError):
    """Monthly refiner spend cap has been reached."""

    def __init__(
        self, message: str, current_spend: float = 0.0, limit: float = 0.0,
        **context: object,
    ) -> None:
        self.current_spend = current_spend
        self.limit = limit
        super().__init__(message, current_spend=current_spend, limit=limit, **context)


class NoFeedbackError(AutoRefineError):
    """Refinement triggered with insufficient feedback data."""


class RollbackError(AutoRefineError):
    """Attempted rollback to a nonexistent or invalid version."""

    def __init__(
        self, message: str, version: int | None = None, **context: object,
    ) -> None:
        self.version = version
        super().__init__(message, version=version, **context)


class SpendCapExceeded(AutoRefineError):
    """The organization's monthly spend cap has been reached."""

    def __init__(
        self, message: str, current_spend: float = 0.0,
        cap: float = 0.0, **context: object,
    ) -> None:
        self.current_spend = current_spend
        self.cap = cap
        super().__init__(message, current_spend=current_spend, cap=cap, **context)


class CloudAuthError(AutoRefineError):
    """AutoRefine Cloud API key is invalid, expired, or revoked."""

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message, **context)
