# Security

AutoRefine is designed for production use with multiple layers of security.

## Multi-tenant isolation

Use the `namespace` parameter to isolate tenants sharing a database:

```python
tenant_a = AutoRefine(api_key=KEY, namespace="tenant_a")
tenant_b = AutoRefine(api_key=KEY, namespace="tenant_b")
```

All prompt_keys, interactions, and feedback are prefixed with the namespace.
Tenant A cannot read or write tenant B's data.

```bash
AUTOREFINE_NAMESPACE=tenant_123
```

## Encryption at rest

When `encryption_key` is set, prompt text and feedback content are encrypted
using Fernet (AES-128-CBC) before being written to the JSON or SQLite store.

```bash
# Generate a key
python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'

# Set it
AUTOREFINE_ENCRYPTION_KEY=your-fernet-key-here
```

```python
client = AutoRefine(
    api_key="sk-...",
    encryption_key="your-fernet-key-here",
)
```

**Encrypted fields**: `system_prompt`, `response_text`, `comment`, `correction`.

Requires: `pip install cryptography`

## API key scrubbing

API keys are automatically scrubbed from all Interaction objects **before storage**.
Patterns detected and replaced with `[KEY_REDACTED]`:

- OpenAI keys (`sk-...`)
- Anthropic keys (`sk-ant-...`)
- Generic API keys and tokens
- Bearer tokens

This means even if your users accidentally paste an API key in a chat message,
it won't persist in the store.

## PII scrubbing

Enabled by default (`pii_scrub_enabled=True`). Before interaction data reaches
the refiner model, these patterns are redacted:

| Pattern | Example | Replacement |
|---------|---------|-------------|
| Email | alice@example.com | `[EMAIL_REDACTED]` |
| Phone | 555-123-4567 | `[PHONE_REDACTED]` |
| SSN | 123-45-6789 | `[SSN_REDACTED]` |
| Credit card | 4111-1111-1111-1111 | `[CREDIT_CARD_REDACTED]` |
| IP address | 192.168.1.1 | `[IP_ADDRESS_REDACTED]` |
| API key | sk-abc123... | `[API_KEY_REDACTED]` |

Add custom patterns:

```python
import re
from autorefine.pii_scrubber import PIIScrubber

scrubber = PIIScrubber(custom_patterns=[
    ("MRN", re.compile(r"MRN-\d{8}")),
])
client = AutoRefine(api_key="sk-...", pii_scrubber=scrubber)
```

## Dashboard security

### Authentication

Set a password to require authentication:

```bash
AUTOREFINE_DASHBOARD_PASSWORD=my-secret-password
```

The dashboard supports both:

- **Session tokens**: POST to `/api/auth/login` with `{"password": "..."}`, get a bearer token
- **Basic auth**: For backward compatibility and simple scripts

### CORS

By default, the dashboard only accepts same-origin requests. Configure
allowed origins explicitly:

```bash
AUTOREFINE_CORS_ORIGINS=https://myapp.com,https://admin.myapp.com
```

### Rate limiting

API endpoints are rate-limited to 10 requests/second per IP by default:

```bash
AUTOREFINE_DASHBOARD_RATE_LIMIT=20
```

## GDPR compliance

### Data export (Article 20 — Portability)

```python
data = client.export_data("default")
# Returns all interactions, feedback, and prompt versions as a dict
```

### Data deletion (Article 17 — Right to erasure)

```python
deleted = client.delete_data("default")
print(f"Deleted {deleted['interactions']} interactions")
```

### Data retention

```bash
AUTOREFINE_RETENTION_DAYS=90
```

Old data is eligible for automatic purging after the retention period.

## Feedback noise filtering

AutoRefine filters malicious or noisy feedback before it reaches the refiner:

- **Rage-click detection**: 5+ negatives in 2 minutes from the same user
- **Contradiction removal**: Same user giving positive and negative on the same response
- **Outlier down-weighting**: One user responsible for >50% of negatives

## Security checklist for production

- [ ] Set `AUTOREFINE_DASHBOARD_PASSWORD`
- [ ] Set `AUTOREFINE_ENCRYPTION_KEY` if storing sensitive data
- [ ] Set `AUTOREFINE_NAMESPACE` if sharing a database across tenants
- [ ] Set `AUTOREFINE_CORS_ORIGINS` to your domain(s)
- [ ] Set `AUTOREFINE_COST_LIMIT_MONTHLY` to prevent budget overruns
- [ ] Set `AUTOREFINE_PII_SCRUB_ENABLED=true` (default)
- [ ] Use `storage_backend=sqlite` or `postgres` (not `json`) for production
- [ ] Never commit `.env` files to version control
- [ ] Rotate API keys regularly
