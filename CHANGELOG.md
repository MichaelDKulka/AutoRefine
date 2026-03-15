# Changelog

All notable changes to AutoRefine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-15

### Added

- **Core SDK**: `AutoRefine` client with `chat()`, `complete()`, `stream()`, `feedback()` methods
- **Async support**: `AsyncAutoRefine` client mirroring the sync API with native async providers
- **Interceptor**: Invisible middleware that swaps in refined prompts and logs interactions
- **Feedback collector**: Batched feedback ingestion with normalization (thumbs up/down, corrections, custom scores)
- **Refiner**: Surgical meta-prompt that patches gaps in system prompts based on feedback patterns
- **A/B testing**: Welch's t-test significance testing with auto-promote/reject (no scipy dependency)
- **Providers**: OpenAI, Anthropic, Mistral, Ollama with per-model pricing and native async
- **Storage backends**: JSON (zero-config), SQLite (WAL mode), PostgreSQL (asyncpg with connection pooling)
- **PII scrubber**: Regex-based redaction of emails, phones, SSNs, API keys before refinement
- **Feedback filter**: Rage-click detection, contradiction removal, outlier user down-weighting
- **Prompt change notifications**: Webhook and callback alerts on version changes
- **Cost tracker**: Unified pricing table (29 models), budget enforcement, spend breakdowns
- **Analytics**: Improvement curves, refinement effectiveness, failure pattern extraction, ROI reports
- **Web dashboard**: FastAPI app with dark theme, Chart.js charts, diff viewer, A/B test controls
- **Feedback widget**: Embeddable HTML/JS widget (minimal/standard/detailed styles)
- **CLI**: `autorefine init`, `dashboard`, `prompts list/show/diff/rollback`, `stats`, `export`, `reset`
- **`@autorefine` decorator**: Wrap any function with one line
- **Error handling**: Classified provider errors (auth/rate-limit/network/malformed), exponential backoff retry
- **Type safety**: Full type annotations, `py.typed` marker, Pydantic v2 models with validators

### Infrastructure

- pytest test suite (284 tests across 11 files)
- GitHub Actions CI (Python 3.9-3.12, ruff, mypy)
- PyPI publish workflow on tag push
