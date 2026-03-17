# Getting Started

This guide walks you through installing AutoRefine and using every major feature, including the analytics dashboard.

## Prerequisites

- Python 3.9+
- An API key for your LLM provider (OpenAI, Anthropic, etc.)
- An Anthropic API key for the refiner (Claude) -- required for automatic prompt improvement

## Step 1: Install

```bash
pip install autorefine
```

With optional providers and features:

```bash
pip install autorefine[openai]        # OpenAI provider
pip install autorefine[anthropic]     # Anthropic provider
pip install autorefine[dashboard]     # Web dashboard (FastAPI + uvicorn)
pip install autorefine[all]           # Everything
```

For local development from source:

```bash
git clone https://github.com/upwell-solutions/autorefine.git
cd autorefine
pip install -e ".[all]"
```

## Step 2: Create a client

```python
from autorefine import AutoRefine

client = AutoRefine(
    api_key="sk-your-openai-key",
    model="gpt-4o",
    refiner_key="sk-ant-your-claude-key",
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

Wire this to your app's UI -- a thumbs up/down button, a rating widget, or a correction form:

```python
# Thumbs up
client.feedback(response.id, "thumbs_up")

# Thumbs down with a comment
client.feedback(response.id, "thumbs_down", comment="Didn't offer a refund")

# User-provided better answer
client.feedback(response.id, "correction", comment="You should proactively offer a refund")
```

## Step 6: Watch it improve

After 10 feedback signals (configurable via `refine_threshold`), AutoRefine automatically:

1. Gathers the feedback + the interactions that triggered it
2. Filters noise (rage-clicks, contradictions) and scrubs PII
3. Builds a detailed meta-prompt with evidence and analysis
4. Sends it to the refiner LLM (Claude) for surgical prompt revision
5. The new prompt either enters an A/B test or is promoted directly
6. All future calls use the improved prompt

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

---

## Structured Feedback with Dimensions

For precise, multi-axis refinement, define quality dimensions at client creation:

```python
client = AutoRefine(
    api_key="sk-...",
    model="gpt-4o",
    refiner_key="sk-ant-...",
    auto_learn=True,
    feedback_dimensions={
        "accuracy": {
            "description": "Was the factual content correct?",
            "weight": 2.0,
            "refinement_priority": "high",
        },
        "reasoning": {
            "description": "Was the chain of reasoning logical and complete?",
            "weight": 1.5,
        },
        "calibration": {
            "description": "Did stated confidence match actual probability?",
            "weight": 1.5,
            "refinement_priority": "high",
        },
        "tone": {
            "description": "Was the response appropriately professional?",
            "weight": 0.5,
            "refinement_priority": "low",
        },
    },
)
```

Then provide per-dimension scores with feedback:

```python
resp = client.complete("You are a prediction bot.", "Will it rain tomorrow?")

client.feedback(
    resp.id,
    signal="negative",
    dimensions={
        "accuracy": -1.0,       # Wrong prediction
        "reasoning": 0.5,       # Good reasoning despite wrong answer
        "calibration": -0.8,    # Overconfident
        "tone": 1.0,            # Professional tone
    },
    comment="Prediction was wrong, confidence was too high",
    context={
        "predicted": "Rain, 85% chance",
        "actual": "Clear skies",
    },
)
```

The refiner receives a per-dimension breakdown showing exactly which quality axes need improvement, making its prompt revisions far more targeted.

Legacy feedback (`client.feedback(resp.id, "thumbs_up")`) still works unchanged.

---

## Refinement Directives

Inject domain knowledge and hard constraints the refiner must respect:

```python
client.set_refinement_directives(
    directives=[
        "Never remove the instruction to cite sources.",
        "When accuracy and tone conflict, always prioritise accuracy.",
        "The model should say 'I don't know' rather than guess.",
    ],
    domain_context=(
        "This is a medical information bot for nurses. Users are "
        "professionals who need precise, evidence-based answers."
    ),
    preserve_behaviors=[
        "The structured format (Summary, Evidence, Caveats) gets positive feedback.",
        "Citation of clinical guidelines is consistently praised.",
    ],
)
```

Directives persist in the store across restarts. Update with:

```python
client.update_refinement_directives(
    add_directives=["Always include dosage ranges when discussing medications."],
    remove_directives=["The model should say 'I don't know' rather than guess."],
)
```

---

## Outcome-Based Feedback

For applications where ground truth arrives after a delay:

```python
# At prediction time
resp = client.complete("", "Will Team A win tonight?")

# Hours later, when the result is known
client.report_outcome(
    response_id=resp.id,
    outcome={
        "predicted": "Team A wins by 7+",
        "actual": "Team B wins by 3",
        "correct": False,
    },
    context={
        "sport": "basketball",
        "key_factor_missed": "Starting player was injured",
    },
)
```

AutoRefine automatically translates outcomes into dimensional scores and feeds them into the refinement pipeline. Confidence is extracted from the model's response text (e.g. "85% chance") to score calibration automatically.

---

## Custom Feedback Provider

For apps that collect feedback through their own UI:

```python
from autorefine import AutoRefine, FeedbackProvider

class WebFeedback(FeedbackProvider):
    def __init__(self, feedback_queue):
        self._queue = feedback_queue

    def get_feedback(self, response_id: str, response_text: str) -> str:
        return self._queue.wait_for(response_id, timeout=300)

client = AutoRefine(
    api_key="sk-...",
    refiner_key="sk-ant-...",
    auto_learn=True,
    feedback_provider=WebFeedback(my_queue),
)

resp = client.complete("You are helpful.", "How do I reset my password?")
client.collect_feedback(resp)  # Calls your provider, records the result
```

---

## Dashboard

AutoRefine includes a web dashboard for monitoring prompt performance in real time.

### Install dashboard dependencies

```bash
pip install autorefine[dashboard]
```

### Launch the dashboard

**From your application:**

```python
client.start_dashboard(port=8787)
# Open http://localhost:8787 in your browser
```

**Standalone (for viewing existing data):**

```python
from autorefine.storage.json_store import JSONStore
from autorefine.dashboard.server import run_dashboard

store = JSONStore()  # Uses ~/.autorefine/store.json
run_dashboard(store, prompt_key="default", port=8787)
```

### Dashboard panels

The dashboard at `http://localhost:8787` has four panels:

**1. Improvement Curve** -- A Chart.js chart showing your prompt's feedback score trending over time. Displays score-per-version (bar chart) and daily average score (line chart). This is your primary indicator that AutoRefine is working.

**2. Active Prompt** -- Shows the full text of the current system prompt with its version number and changelog. Includes a diff viewer (green = added, red = removed) showing what changed in the last refinement. A rollback dropdown lets you instantly revert to any previous version.

**3. Recent Feedback** -- A live feed of user feedback signals, auto-refreshing every 15 seconds. Each entry shows the feedback type (positive/negative/correction), score, timestamp, and user comment. Filter by type using the dropdown.

**4. A/B Test Status** -- When a candidate prompt is being tested, this panel shows champion vs candidate scores, sample counts, and traffic split. Promote or reject buttons let you override the statistical test manually.

### Password protection

```bash
export AUTOREFINE_DASHBOARD_PASSWORD=mysecretpassword
```

Or in code:

```python
run_dashboard(store, password="mysecretpassword", port=8787)
```

### Embeddable feedback widget

Generate an HTML widget for your web pages:

```python
html = client.get_widget_html(response_id=resp.id, style="standard")
# Inject into your page template
```

Three styles: `"minimal"` (icon buttons only), `"standard"` (buttons + comment box), `"detailed"` (buttons + comment + category tags). The widget POSTs feedback to the dashboard API.

---

## Configuration Reference

All settings can be set via constructor, environment variables (`AUTOREFINE_` prefix), or `.env` file.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `api_key` | `AUTOREFINE_API_KEY` | `""` | LLM provider API key |
| `model` | `AUTOREFINE_MODEL` | `gpt-3.5-turbo` | Model for user-facing calls |
| `refiner_key` | `AUTOREFINE_REFINER_KEY` | `""` | API key for the refiner LLM |
| `refiner_model` | `AUTOREFINE_REFINER_MODEL` | `claude-sonnet-4-20250514` | Refiner model |
| `auto_learn` | `AUTOREFINE_AUTO_LEARN` | `false` | Auto-refine when threshold is reached |
| `refine_threshold` | `AUTOREFINE_REFINE_THRESHOLD` | `10` | Feedback count before auto-refinement |
| `storage_backend` | `AUTOREFINE_STORAGE_BACKEND` | `json` | `json`, `sqlite`, or `postgres` |
| `ab_test_split` | `AUTOREFINE_AB_TEST_SPLIT` | `0.2` | Candidate traffic fraction (0 = skip A/B) |
| `cost_limit_monthly` | `AUTOREFINE_COST_LIMIT_MONTHLY` | `25.0` | Monthly USD cap for refiner calls |
| `dashboard_port` | `AUTOREFINE_DASHBOARD_PORT` | `8787` | Dashboard server port |
| `pii_scrub_enabled` | `AUTOREFINE_PII_SCRUB_ENABLED` | `true` | Redact PII before sending to refiner |

### Storage backends

```python
# JSON (default -- zero config, good for dev)
client = AutoRefine(api_key="sk-...", storage_backend="json")

# SQLite (production single-server)
client = AutoRefine(api_key="sk-...", storage_backend="sqlite")

# PostgreSQL (production multi-server)
pip install autorefine[postgres]
client = AutoRefine(
    api_key="sk-...",
    storage_backend="postgres",
    database_url="postgresql://user:pass@host:5432/dbname",
)
```

---

## API Quick Reference

| Method | Description |
|--------|-------------|
| `complete(system, prompt)` | Single-prompt completion |
| `chat(system, messages)` | Multi-turn chat |
| `stream(system, messages)` | Streaming chat |
| `feedback(response_id, signal, ...)` | Record feedback (with optional dimensions/context) |
| `collect_feedback(response)` | Collect feedback via a FeedbackProvider |
| `report_outcome(response_id, outcome, ...)` | Report ground-truth outcome |
| `set_refinement_directives(...)` | Set hard constraints for the refiner |
| `update_refinement_directives(...)` | Merge updates to existing directives |
| `set_system_prompt(prompt)` | Manually set a prompt version |
| `get_active_prompt()` | Get the current active prompt |
| `rollback(version)` | Revert to a previous version |
| `refine_now()` | Manually trigger a refinement cycle |
| `start_dashboard(port)` | Launch the web dashboard |
| `get_widget_html(response_id, style)` | Get embeddable feedback widget HTML |
| `get_analytics(days)` | Get performance analytics snapshot |
| `health_check()` | Verify all components are reachable |

## Next steps

- [Feedback Types](feedback-types.md) -- all the ways to collect feedback
- [Configuration](configuration.md) -- every setting explained
- [Deployment](deployment.md) -- production setup with SQLite or PostgreSQL
- [API Reference](api-reference.md) -- full method documentation
- [Security](security.md) -- PII scrubbing, encryption, dashboard auth
