"""Tests for AutoRefineSettings — env vars, .env file, constructor overrides,
and precedence order.
"""

from __future__ import annotations

import pytest

from autorefine.config import AutoRefineSettings


class TestDefaults:
    """Verify sensible defaults are set."""

    def test_default_model(self):
        cfg = AutoRefineSettings()
        assert cfg.model == "gpt-3.5-turbo"

    def test_default_refiner_model(self):
        cfg = AutoRefineSettings()
        assert cfg.refiner_model == "claude-sonnet-4-20250514"

    def test_default_storage_backend(self):
        cfg = AutoRefineSettings()
        assert cfg.storage_backend == "json"

    def test_default_refine_threshold(self):
        cfg = AutoRefineSettings()
        assert cfg.refine_threshold == 10

    def test_default_refine_batch_size(self):
        cfg = AutoRefineSettings()
        assert cfg.refine_batch_size == 50

    def test_default_ab_test_split(self):
        cfg = AutoRefineSettings()
        assert cfg.ab_test_split == 0.2

    def test_default_cost_limit(self):
        cfg = AutoRefineSettings()
        assert cfg.cost_limit_monthly == 25.0

    def test_default_auto_learn_is_false(self):
        cfg = AutoRefineSettings()
        assert cfg.auto_learn is False

    def test_default_pii_scrub_enabled(self):
        cfg = AutoRefineSettings()
        assert cfg.pii_scrub_enabled is True

    def test_default_feedback_filter_enabled(self):
        cfg = AutoRefineSettings()
        assert cfg.feedback_filter_enabled is True

    def test_default_dashboard_port(self):
        cfg = AutoRefineSettings()
        assert cfg.dashboard_port == 8787

    def test_default_retention_days(self):
        cfg = AutoRefineSettings()
        assert cfg.retention_days == 90


class TestConstructorOverrides:
    """Settings passed directly to the constructor take effect."""

    def test_api_key(self):
        cfg = AutoRefineSettings(api_key="sk-test-123")
        assert cfg.api_key == "sk-test-123"

    def test_model(self):
        cfg = AutoRefineSettings(model="gpt-4o")
        assert cfg.model == "gpt-4o"

    def test_refine_threshold(self):
        cfg = AutoRefineSettings(refine_threshold=5)
        assert cfg.refine_threshold == 5

    def test_storage_backend(self):
        cfg = AutoRefineSettings(storage_backend="sqlite")
        assert cfg.storage_backend == "sqlite"

    def test_cost_limit(self):
        cfg = AutoRefineSettings(cost_limit_monthly=100.0)
        assert cfg.cost_limit_monthly == 100.0

    def test_ab_test_split(self):
        cfg = AutoRefineSettings(ab_test_split=0.5)
        assert cfg.ab_test_split == 0.5

    def test_auto_learn(self):
        cfg = AutoRefineSettings(auto_learn=True)
        assert cfg.auto_learn is True

    def test_webhook_url(self):
        cfg = AutoRefineSettings(webhook_url="https://hooks.example.com/test")
        assert cfg.webhook_url == "https://hooks.example.com/test"


class TestEnvironmentVariables:
    """Settings loaded from env vars (prefixed with AUTOREFINE_)."""

    def test_env_api_key(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_API_KEY", "sk-from-env")
        cfg = AutoRefineSettings()
        assert cfg.api_key == "sk-from-env"

    def test_env_model(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_MODEL", "gpt-4o-mini")
        cfg = AutoRefineSettings()
        assert cfg.model == "gpt-4o-mini"

    def test_env_refine_threshold(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_REFINE_THRESHOLD", "42")
        cfg = AutoRefineSettings()
        assert cfg.refine_threshold == 42

    def test_env_auto_learn(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_AUTO_LEARN", "true")
        cfg = AutoRefineSettings()
        assert cfg.auto_learn is True

    def test_env_storage_backend(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_STORAGE_BACKEND", "sqlite")
        cfg = AutoRefineSettings()
        assert cfg.storage_backend == "sqlite"

    def test_env_cost_limit(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_COST_LIMIT_MONTHLY", "99.50")
        cfg = AutoRefineSettings()
        assert cfg.cost_limit_monthly == 99.50

    def test_env_dashboard_port(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_DASHBOARD_PORT", "9090")
        cfg = AutoRefineSettings()
        assert cfg.dashboard_port == 9090


class TestDotEnvFile:
    """Settings loaded from a .env file."""

    def test_reads_dotenv(self, monkeypatch, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text(
            "AUTOREFINE_API_KEY=sk-from-dotenv\n"
            "AUTOREFINE_MODEL=from-dotenv\n"
            "AUTOREFINE_REFINE_THRESHOLD=7\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        cfg = AutoRefineSettings(_env_file=str(dotenv))
        assert cfg.api_key == "sk-from-dotenv"
        assert cfg.model == "from-dotenv"
        assert cfg.refine_threshold == 7


class TestPrecedence:
    """Constructor > env var > .env > default."""

    def test_constructor_beats_env(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_MODEL", "from-env")
        cfg = AutoRefineSettings(model="from-constructor")
        assert cfg.model == "from-constructor"

    def test_env_beats_default(self, monkeypatch):
        monkeypatch.setenv("AUTOREFINE_REFINE_THRESHOLD", "99")
        cfg = AutoRefineSettings()
        assert cfg.refine_threshold == 99

    def test_constructor_beats_dotenv(self, monkeypatch, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("AUTOREFINE_API_KEY=from-dotenv\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        cfg = AutoRefineSettings(_env_file=str(dotenv), api_key="from-constructor")
        assert cfg.api_key == "from-constructor"


class TestValidation:
    """Settings validation."""

    def test_invalid_storage_backend_raises(self):
        with pytest.raises(ValueError, match="storage_backend"):
            AutoRefineSettings(storage_backend="redis")

    def test_refine_threshold_min_1(self):
        with pytest.raises(ValueError):
            AutoRefineSettings(refine_threshold=0)

    def test_ab_test_split_range(self):
        with pytest.raises(ValueError):
            AutoRefineSettings(ab_test_split=1.5)

    def test_dashboard_port_range(self):
        with pytest.raises(ValueError):
            AutoRefineSettings(dashboard_port=0)

    def test_cost_limit_non_negative(self):
        with pytest.raises(ValueError):
            AutoRefineSettings(cost_limit_monthly=-1.0)

    def test_extra_fields_ignored(self):
        # Should not raise — extra="ignore" in model_config
        cfg = AutoRefineSettings(nonexistent_field="whatever")
        assert not hasattr(cfg, "nonexistent_field")


class TestDetectProvider:
    """Provider auto-detection from model name."""

    def test_gpt_models(self):
        assert AutoRefineSettings(model="gpt-4o").detect_provider() == "openai"
        assert AutoRefineSettings(model="gpt-3.5-turbo").detect_provider() == "openai"

    def test_o_series(self):
        assert AutoRefineSettings(model="o1").detect_provider() == "openai"
        assert AutoRefineSettings(model="o3-mini").detect_provider() == "openai"

    def test_claude_models(self):
        assert AutoRefineSettings(model="claude-sonnet-4-20250514").detect_provider() == "anthropic"
        assert AutoRefineSettings(model="claude-3-haiku-20240307").detect_provider() == "anthropic"

    def test_mistral_models(self):
        assert AutoRefineSettings(model="mistral-large-latest").detect_provider() == "mistral"

    def test_ollama_models(self):
        assert AutoRefineSettings(model="llama3.1").detect_provider() == "ollama"
        assert AutoRefineSettings(model="gemma2").detect_provider() == "ollama"

    def test_explicit_provider_overrides(self):
        cfg = AutoRefineSettings(model="my-custom-model", provider="ollama")
        assert cfg.detect_provider() == "ollama"

    def test_unknown_defaults_to_openai(self):
        assert AutoRefineSettings(model="some-unknown-model").detect_provider() == "openai"


class TestStorePath:
    """get_store_path() resolution."""

    def test_default_path(self):
        cfg = AutoRefineSettings()
        path = cfg.get_store_path()
        assert "autorefine" in path
        assert path.endswith("store.json")

    def test_custom_path(self):
        cfg = AutoRefineSettings(store_path="/tmp/my-store.json")
        assert cfg.get_store_path() == "/tmp/my-store.json"
