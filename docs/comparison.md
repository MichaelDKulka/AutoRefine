# Comparison

How AutoRefine compares to other prompt management and optimization tools.

## Feature matrix

| Feature | AutoRefine | DSPy | PromptLayer | Humanloop | LangSmith |
|---------|-----------|------|-------------|-----------|-----------|
| **Automatic prompt refinement** | Yes -- from user feedback | Requires compilation step | No | No | No |
| **Zero code changes** | Yes -- drop-in interceptor | No -- new abstractions | Logging only | Logging only | Logging only |
| **A/B testing** | Built-in, Welch's t-test | No | Manual | Manual | No |
| **PII scrubbing** | Built-in (regex) | No | No | No | No |
| **Feedback noise filtering** | Built-in (rage-clicks, outliers) | No | No | No | No |
| **Provider-agnostic** | OpenAI, Anthropic, Mistral, Ollama | OpenAI-centric | OpenAI-centric | Multi | Multi |
| **Self-hosted** | Yes (default) | Yes | No (SaaS) | No (SaaS) | Partial |
| **Open source** | Yes (MIT) | Yes (MIT) | Partial | No | Partial |
| **Prompt versioning** | Built-in with rollback | No | Yes | Yes | Yes |
| **Prompt change webhooks** | Built-in | No | No | No | No |
| **Cost tracking** | Per-call USD, 29 models | No | No | Yes | Yes |
| **Storage options** | JSON, SQLite, PostgreSQL | None | Cloud-only | Cloud-only | Cloud-only |
| **Web dashboard** | Built-in (FastAPI) | No | Yes (SaaS) | Yes (SaaS) | Yes (SaaS) |
| **Embeddable widget** | HTML/JS, 3 styles | No | No | No | No |
| **CLI** | 9 commands | No | No | No | Yes |
| **Async support** | Native (asyncio) | No | No | Yes | Yes |
| **Pricing** | Free (OSS) + LLM costs | Free (OSS) | $29+/mo | $79+/mo | $39+/mo |

## When to use what

### Use AutoRefine when...

- You want prompts to improve from real user feedback automatically
- You need self-hosted, auditable prompt versioning
- You want A/B testing built into the refinement loop
- You're cost-conscious and want budget guardrails
- Privacy matters -- you want PII scrubbed before refinement

### Use DSPy when...

- You're doing academic research on prompt optimization
- You have a dataset of labeled examples to compile against
- You want fine-grained control over prompt optimization algorithms

### Use PromptLayer / Humanloop when...

- You want a managed SaaS with a team collaboration UI
- You need non-technical team members to edit prompts
- You're already committed to their ecosystem

### Use LangSmith when...

- You're already using LangChain
- You primarily need tracing and debugging
- You want dataset-driven evaluation
