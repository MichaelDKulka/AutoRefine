# Changelog

## [0.1.0] - 2026-03-15

### Added

- **Core SDK**: `AutoRefine` client with `chat()`, `complete()`, `stream()`, `feedback()`
- **Async support**: `AsyncAutoRefine` with native async providers
- **Interceptor**: Invisible middleware that swaps in refined prompts
- **Feedback collector**: Batched ingestion with normalization
- **Refiner**: Surgical meta-prompt that patches prompt gaps from feedback
- **A/B testing**: Welch's t-test significance testing (no scipy)
- **Providers**: OpenAI, Anthropic, Mistral, Ollama with per-model pricing
- **Storage**: JSON (zero-config), SQLite (WAL), PostgreSQL (asyncpg)
- **PII scrubber**: Regex redaction of emails, phones, SSNs, API keys
- **Feedback filter**: Rage-click detection, contradiction removal, outlier down-weighting
- **Cost tracker**: Unified pricing table (29 models), budget enforcement
- **Analytics**: Improvement curves, failure patterns, ROI reports
- **Dashboard**: FastAPI web UI with Chart.js, diff viewer, A/B controls
- **Widget**: Embeddable HTML/JS feedback widget (3 styles)
- **CLI**: `autorefine init`, `dashboard`, `prompts`, `stats`, `export`, `reset`
- **`@autorefine` decorator**: Wrap any function with one line
- **Error handling**: Classified provider errors, exponential backoff retry

For the complete changelog, see the [CHANGELOG.md on GitHub](https://github.com/upwell-solutions/autorefine/blob/main/CHANGELOG.md).
