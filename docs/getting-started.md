# Getting Started

This guide walks you through setting up AutoRefine from scratch in under 5 minutes.

## Prerequisites

- Python 3.9+
- An API key for your LLM provider (OpenAI, Anthropic, etc.)
- Optionally, an Anthropic API key for the refiner (Claude)

## Step 1: Install

```bash
pip install autorefine[openai]
```

Or use the CLI to set up everything interactively:

```bash
pip install autorefine
autorefine init
```

## Step 2: Create a client

```python
from autorefine import AutoRefine

client = AutoRefine(
    api_key="sk-your-openai-key",
    model="gpt-4o",
    refiner_key="sk-ant-your-claude-key",  # optional
    auto_learn=True,                        # auto-refine on feedback
)
```

## Step 3: Set your initial prompt

```python
client.set_system_prompt(
    "You are a helpful customer support agent. Be empathetic, "
    "precise, and always offer to escalate if the issue is complex."
)
```

## Step 4: Use it like normal

```python
response = client.chat(
    "You are a helpful customer support agent.",
    [{"role": "user", "content": "I was charged twice!"}],
)
print(response.text)
print(f"Response ID: {response.id}")
```

## Step 5: Collect feedback

Wire this to your app's UI — a thumbs up/down button, a rating widget, or a correction form:

```python
# Thumbs up
client.feedback(response.id, "thumbs_up")

# Thumbs down with a comment
client.feedback(response.id, "thumbs_down", comment="Didn't offer a refund")

# User-provided better answer
client.feedback(response.id, "correction", comment="You should proactively offer a refund")
```

## Step 6: Watch it improve

After `refine_threshold` feedback signals (default: 20), AutoRefine automatically:

1. Gathers the feedback + the interactions that triggered it
2. Sends them to Claude with a detailed meta-prompt
3. Claude surgically patches your system prompt
4. The new prompt enters an A/B test
5. If it wins, it's promoted automatically

Check the current state:

```python
# Current active prompt
active = client.get_active_prompt()
print(f"v{active.version}: {active.system_prompt}")

# Full history
for pv in client.get_prompt_history():
    print(f"v{pv.version} {'(active)' if pv.is_active else ''}: {pv.changelog}")

# Analytics
snap = client.get_analytics(days=7)
print(f"Score: {snap.average_score:+.2f}, Feedback: {snap.total_feedback}")
```

## Step 7: Launch the dashboard

```python
client.start_dashboard(port=8787)
# Open http://localhost:8787
```

Or from the CLI:

```bash
autorefine dashboard --port 8787
```

## Next steps

- [Feedback Types](feedback-types.md) — all the ways to collect feedback
- [Configuration](configuration.md) — every setting explained
- [Deployment](deployment.md) — production setup with SQLite or PostgreSQL
- [API Reference](api-reference.md) — full method documentation
