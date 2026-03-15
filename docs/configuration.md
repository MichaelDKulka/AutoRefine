# Configuration Reference

All settings can be set via constructor arguments, environment variables
(prefixed with `AUTOREFINE_`), or a `.env` file.

**Precedence**: constructor args > env vars > `.env` file > defaults.

## Settings table

| Setting | Env var | Default | Description |
|---------|---------|---------|-------------|
| `api_key` | `AUTOREFINE_API_KEY` | `""` | API key for the primary LLM provider |
| `model` | `AUTOREFINE_MODEL` | `gpt-3.5-turbo` | Model identifier (e.g. `gpt-4o`, `claude-sonnet-4-20250514`) |
| `provider` | `AUTOREFINE_PROVIDER` | `""` (auto) | Provider name. Auto-detected from model name if empty |
| `refiner_key` | `AUTOREFINE_REFINER_KEY` | `""` | API key for the refiner model. Required for refinement |
| `refiner_model` | `AUTOREFINE_REFINER_MODEL` | `claude-sonnet-4-20250514` | Model used for prompt refinement |
| `refiner_provider` | `AUTOREFINE_REFINER_PROVIDER` | `anthropic` | Provider for the refiner model |
| `auto_learn` | `AUTOREFINE_AUTO_LEARN` | `false` | Auto-trigger refinement when threshold is met |
| `refine_threshold` | `AUTOREFINE_REFINE_THRESHOLD` | `20` | Minimum feedback count before auto-refinement |
| `refine_batch_size` | `AUTOREFINE_REFINE_BATCH_SIZE` | `50` | Max feedback signals per refinement call |
| `max_prompt_versions` | `AUTOREFINE_MAX_PROMPT_VERSIONS` | `50` | Max versions retained per prompt_key |
| `ab_test_split` | `AUTOREFINE_AB_TEST_SPLIT` | `0.2` | Fraction of traffic for A/B candidate (0.0 = skip A/B) |
| `ab_test_min_interactions` | `AUTOREFINE_AB_TEST_MIN_INTERACTIONS` | `100` | Min interactions per variant before auto-resolve |
| `storage_backend` | `AUTOREFINE_STORAGE_BACKEND` | `json` | Backend: `json`, `sqlite`, or `postgres` |
| `store_path` | `AUTOREFINE_STORE_PATH` | `~/.autorefine/store.json` | File path for json/sqlite backends |
| `database_url` | `AUTOREFINE_DATABASE_URL` | `""` | PostgreSQL connection string |
| `dashboard_port` | `AUTOREFINE_DASHBOARD_PORT` | `8787` | Dashboard web UI port |
| `dashboard_password` | `AUTOREFINE_DASHBOARD_PASSWORD` | `""` | Basic auth password for dashboard |
| `cost_limit_monthly` | `AUTOREFINE_COST_LIMIT_MONTHLY` | `25.0` | Monthly USD cap for refiner calls |
| `retention_days` | `AUTOREFINE_RETENTION_DAYS` | `90` | Days before old data is eligible for purging |
| `pii_scrub_enabled` | `AUTOREFINE_PII_SCRUB_ENABLED` | `true` | Redact PII before sending to refiner |
| `feedback_filter_enabled` | `AUTOREFINE_FEEDBACK_FILTER_ENABLED` | `true` | Filter noisy feedback (rage-clicks, outliers) |
| `webhook_url` | `AUTOREFINE_WEBHOOK_URL` | `""` | URL for prompt-change webhook notifications |

## Example `.env` file

```bash
AUTOREFINE_API_KEY=sk-...
AUTOREFINE_MODEL=gpt-4o
AUTOREFINE_REFINER_KEY=sk-ant-...
AUTOREFINE_AUTO_LEARN=true
AUTOREFINE_STORAGE_BACKEND=sqlite
AUTOREFINE_COST_LIMIT_MONTHLY=25.0
AUTOREFINE_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## Provider auto-detection

When `provider` is empty, AutoRefine guesses from the model name:

| Model pattern | Detected provider |
|---------------|-------------------|
| `gpt-*`, `o1*`, `o3*`, `o4*` | `openai` |
| `claude-*` | `anthropic` |
| `mistral-*`, `mixtral-*` | `mistral` |
| `llama*`, `gemma*`, `phi*`, `qwen*` | `ollama` |
| (unknown) | `openai` (fallback) |

## Storage backends

| Backend | Best for | Config |
|---------|----------|--------|
| `json` | Development, prototyping | Zero config — writes to `~/.autorefine/store.json` |
| `sqlite` | Single-server production | WAL mode, proper indexes, thread-safe |
| `postgres` | Multi-server, distributed | asyncpg connection pooling, row-level locking |
