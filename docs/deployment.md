# Production Deployment

## Storage backends

### SQLite (recommended for single-server)

```python
client = AutoRefine(
    api_key="sk-...",
    model="gpt-4o",
    storage_backend="sqlite",
    store_path="/data/autorefine.db",
)
```

Or via environment:

```bash
AUTOREFINE_STORAGE_BACKEND=sqlite
AUTOREFINE_STORE_PATH=/data/autorefine.db
```

SQLite uses WAL mode for concurrent reads during writes and includes indexes on all frequently queried columns.

### PostgreSQL (recommended for multi-server)

```bash
pip install autorefine[postgres]
```

```python
client = AutoRefine(
    api_key="sk-...",
    model="gpt-4o",
    storage_backend="postgres",
    database_url="postgresql://user:pass@host:5432/autorefine",
)
```

Features: asyncpg connection pooling, row-level locking on prompt activation, JSONB columns.

## Cost controls

Set a monthly budget to prevent runaway refiner costs:

```python
client = AutoRefine(
    ...,
    cost_limit_monthly=25.0,  # USD
)
```

When the limit is hit, refinement pauses automatically until the next month.

## Security

### PII scrubbing

Enabled by default. User messages and model responses are redacted before reaching the refiner:

- Emails, phone numbers, SSNs
- Credit card numbers, IP addresses
- API keys and tokens

### Dashboard authentication

```python
client = AutoRefine(..., dashboard_password="my-secret")
client.start_dashboard()
```

### Webhook alerts

Get notified on every prompt change:

```bash
AUTOREFINE_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...
```

## Monitoring

### Health check

```python
status = client.health_check()
# {"ok": True, "provider": "ok", "store": "ok", "refiner": "ok"}
```

### ROI reports

```python
from autorefine.analytics import Analytics
analytics = Analytics(store, prompt_key="default")
print(analytics.generate_roi_report(days=30))
```

## Docker

```dockerfile
FROM python:3.12-slim
RUN pip install autorefine[openai,dashboard]
COPY app.py .
CMD ["python", "app.py"]
```

## Multi-prompt namespaces

Use `prompt_key` to maintain independent prompt evolution per use case:

```python
billing = AutoRefine(api_key=KEY, model="gpt-4o", prompt_key="billing")
tech = AutoRefine(api_key=KEY, model="gpt-4o", prompt_key="technical")
```

Each namespace refines independently based on its own feedback.
