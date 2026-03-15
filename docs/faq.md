# FAQ

## General

### What is AutoRefine?

AutoRefine is a Python SDK that sits between your application and the LLM. It intercepts calls, collects user feedback, and automatically rewrites your system prompts to improve quality -- without any code changes on your part.

### How is it different from fine-tuning?

Fine-tuning changes model weights (expensive, slow, requires training data). AutoRefine changes your system prompt (instant, cheap, works with any model). They're complementary -- you can fine-tune the model AND auto-refine the prompt.

### Does it work with any LLM?

Yes. AutoRefine supports OpenAI, Anthropic, Mistral, Ollama, and any OpenAI-compatible API. The refiner defaults to Claude but can use any model.

### What happens if the refiner makes the prompt worse?

Three safeguards:

1. **A/B testing** -- candidates are tested against the current prompt with real traffic. Only statistically significant winners are promoted.
2. **Rollback** -- `client.rollback(version=1)` instantly reverts to any previous version.
3. **Budget limits** -- refinement pauses when the monthly cost cap is hit.

## Technical

### Does AutoRefine add latency?

The interceptor adds <1ms (prompt lookup from local storage). The LLM call itself is unchanged. Refinement runs asynchronously in the background.

### What if the store is down?

The interceptor falls through to the developer's original prompt. No error is raised. The LLM call works as if AutoRefine wasn't there.

### Can I use it without a refiner key?

Yes. Without a refiner key, AutoRefine still logs interactions and feedback. You get analytics, versioning, and rollback -- just no automatic refinement.

### Is my data sent to third parties?

Only during refinement, and only to the refiner model (Claude by default). PII scrubbing is enabled by default to redact sensitive data first. All other data stays in your chosen storage backend (local JSON, SQLite, or your own Postgres).

### How much does refinement cost?

Each refinement cycle uses ~2K-5K tokens on the refiner model. At Claude Sonnet pricing ($3/$15 per 1M tokens), that's roughly $0.01-0.05 per refinement. The default budget cap is $25/month.

## Usage

### How much feedback do I need?

The default threshold is 20 signals. In practice, 20-50 diverse signals produce good refinements. More is better, but there are diminishing returns past ~100.

### Can I have multiple prompts?

Yes. Use `prompt_key` to maintain independent prompt namespaces:

```python
billing = AutoRefine(..., prompt_key="billing")
support = AutoRefine(..., prompt_key="support")
```

Each refines independently.

### Can I use it with async frameworks?

Yes. Use `AsyncAutoRefine`:

```python
from autorefine import AsyncAutoRefine

client = AsyncAutoRefine(api_key="sk-...", model="gpt-4o")
resp = await client.chat("Be helpful.", [{"role": "user", "content": "Hi"}])
await client.feedback(resp.id, "thumbs_up")
```

### How do I deploy to production?

See the [Deployment guide](deployment.md). TL;DR: use SQLite for single-server, PostgreSQL for multi-server, and set `cost_limit_monthly` to control spend.
