"""Centralized configuration for AutoRefine.

All settings can be supplied as constructor arguments, environment variables
(prefixed with ``AUTOREFINE_``), or entries in a ``.env`` file.  Precedence:
constructor args > env vars > ``.env`` > defaults.

Example ``.env``::

    AUTOREFINE_API_KEY=sk-...
    AUTOREFINE_REFINER_KEY=sk-ant-...
    AUTOREFINE_STORAGE_BACKEND=sqlite
    AUTOREFINE_COST_LIMIT_MONTHLY=25.0
"""

from __future__ import annotations

import pathlib

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AutoRefineSettings(BaseSettings):
    """All tuneable parameters with sensible, production-ready defaults.

    Group overview
    --------------
    - **LLM** — primary model used for user-facing calls.
    - **Refiner** — model used to analyse feedback and rewrite prompts.
    - **Learning** — controls when automatic refinement is triggered.
    - **A/B testing** — governs how candidate prompts are validated.
    - **Storage** — persistence backend selection.
    - **Dashboard** — optional web UI.
    - **Cost** — spending guardrails.
    - **Data retention** — how long raw data is kept.
    - **Privacy** — PII scrubbing toggle.
    - **Feedback filtering** — noise-reduction toggle.
    - **Notifications** — prompt-change alerting.
    """

    model_config = {"env_prefix": "AUTOREFINE_", "env_file": ".env", "extra": "ignore"}

    # -- LLM -----------------------------------------------------------------

    api_key: str = Field(
        default="",
        description="API key for the primary LLM provider.",
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="Model identifier for user-facing calls (e.g. 'gpt-4o', 'claude-sonnet-4-20250514').",
    )
    provider: str = Field(
        default="",
        description=(
            "LLM provider name ('openai', 'anthropic', 'mistral', 'ollama'). "
            "Auto-detected from the model name when left empty."
        ),
    )

    # -- Refiner -------------------------------------------------------------

    refiner_key: str = Field(
        default="",
        description="API key for the refiner model. Required for automatic prompt improvement.",
    )
    refiner_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model used by the refiner to analyse feedback and rewrite prompts.",
    )
    refiner_provider: str = Field(
        default="anthropic",
        description="Provider for the refiner model.",
    )

    # -- Learning ------------------------------------------------------------

    auto_learn: bool = Field(
        default=False,
        description=(
            "When True, refinement triggers automatically once enough feedback "
            "accumulates (see refine_threshold). When False, call refine_now() manually."
        ),
    )

    # -- Refinement tuning ---------------------------------------------------

    refine_threshold: int = Field(
        default=20,
        ge=1,
        description=(
            "Minimum number of unprocessed feedback signals required before "
            "an automatic refinement cycle is triggered."
        ),
    )
    refine_batch_size: int = Field(
        default=50,
        ge=1,
        description=(
            "Maximum feedback signals consumed per refinement call. "
            "Larger batches give the refiner more context but cost more tokens."
        ),
    )
    max_prompt_versions: int = Field(
        default=50,
        ge=1,
        description="Maximum prompt versions retained per prompt_key before the oldest are pruned.",
    )

    # -- A/B testing ---------------------------------------------------------

    ab_test_split: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of traffic routed to the candidate prompt during an A/B test. "
            "Set to 0.0 to skip A/B testing and promote candidates immediately."
        ),
    )
    ab_test_min_interactions: int = Field(
        default=100,
        ge=1,
        description="Minimum interactions each variant must accumulate before the test auto-resolves.",
    )

    # -- Storage -------------------------------------------------------------

    storage_backend: str = Field(
        default="json",
        description="Persistence backend: 'json' (zero-config file), 'sqlite' (single-server), or 'postgres' (distributed).",
    )
    store_path: str = Field(
        default="",
        description=(
            "File path for json/sqlite backends. "
            "Defaults to ~/.autorefine/store.json (or .db for sqlite) when empty."
        ),
    )
    database_url: str = Field(
        default="",
        description="PostgreSQL connection string (only used when storage_backend='postgres').",
    )

    @field_validator("storage_backend")
    @classmethod
    def _validate_storage_backend(cls, v: str) -> str:
        """Ensure storage_backend is one of the supported values."""
        allowed = {"json", "sqlite", "postgres"}
        if v not in allowed:
            raise ValueError(f"storage_backend must be one of {allowed}, got '{v}'")
        return v

    # -- Dashboard -----------------------------------------------------------

    dashboard_port: int = Field(
        default=8787,
        ge=1,
        le=65535,
        description="Port for the optional analytics dashboard web UI.",
    )
    dashboard_password: str = Field(
        default="",
        description="Password for basic-auth protection of the dashboard (empty = no auth).",
    )

    # -- Cost ----------------------------------------------------------------

    cost_limit_monthly: float = Field(
        default=25.0,
        ge=0.0,
        description=(
            "Monthly spending cap in USD for refiner API calls. "
            "A CostLimitExceeded exception is raised when the limit is hit."
        ),
    )

    # -- Data retention ------------------------------------------------------

    retention_days: int = Field(
        default=90,
        ge=1,
        description="Interactions, feedback, and cost entries older than this are eligible for purging.",
    )

    # -- Privacy -------------------------------------------------------------

    pii_scrub_enabled: bool = Field(
        default=True,
        description=(
            "When True, user messages and model responses are redacted for PII "
            "(emails, phone numbers, SSNs, API keys, etc.) before being sent to "
            "the refiner model."
        ),
    )

    # -- Feedback filtering --------------------------------------------------

    feedback_filter_enabled: bool = Field(
        default=True,
        description=(
            "When True, noisy feedback (rage-clicks, contradictions, outlier users) "
            "is automatically filtered before refinement."
        ),
    )

    # -- Notifications -------------------------------------------------------

    webhook_url: str = Field(
        default="",
        description=(
            "URL to POST a JSON payload whenever a prompt version changes. "
            "Useful for Slack, PagerDuty, or custom alerting. Empty = disabled."
        ),
    )

    # -- Multi-tenancy -------------------------------------------------------

    namespace: str = Field(
        default="",
        description=(
            "Tenant namespace prefix for all stored data. When set, all "
            "prompt_keys, table names, and JSON keys are prefixed so "
            "multiple tenants can share a database without data leakage."
        ),
    )

    # -- Encryption at rest --------------------------------------------------

    encryption_key: str = Field(
        default="",
        description=(
            "Fernet encryption key for data at rest. When set, prompt text "
            "and feedback content are encrypted before storage. Generate "
            "with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        ),
    )

    # -- Dashboard security --------------------------------------------------

    cors_origins: str = Field(
        default="",
        description=(
            "Comma-separated list of allowed CORS origins for the dashboard API. "
            "Empty = same-origin only. Set to '*' to allow all (not recommended)."
        ),
    )
    dashboard_rate_limit: int = Field(
        default=10,
        ge=1,
        description="Max requests per second to dashboard API endpoints.",
    )

    # -- Derived helpers -----------------------------------------------------

    def get_store_path(self) -> str:
        """Return the resolved store file path, creating parent dirs if needed."""
        if self.store_path:
            return self.store_path
        default = pathlib.Path.home() / ".autorefine" / "store.json"
        default.parent.mkdir(parents=True, exist_ok=True)
        return str(default)

    def detect_provider(self) -> str:
        """Infer the provider name from the model identifier.

        Falls back to ``'openai'`` if no known pattern matches.
        """
        if self.provider:
            return self.provider
        model_lower = self.model.lower()
        if any(k in model_lower for k in ("gpt", "o1", "o3", "davinci", "chatgpt")):
            return "openai"
        if "claude" in model_lower:
            return "anthropic"
        if any(k in model_lower for k in ("mistral", "mixtral", "codestral")):
            return "mistral"
        if any(k in model_lower for k in ("llama", "gemma", "phi", "qwen")):
            return "ollama"
        return "openai"
