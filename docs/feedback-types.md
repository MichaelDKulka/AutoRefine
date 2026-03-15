# Feedback Types

AutoRefine supports explicit feedback (user-initiated) and implicit feedback (behavior-based, coming soon).

## Explicit signals

| Signal | Score | Confidence | Usage |
|--------|-------|-----------|-------|
| `thumbs_up` / `positive` | +1.0 | 1.0 | User liked the response |
| `thumbs_down` / `negative` | -1.0 | 1.0 | User disliked the response |
| `correction` | -0.5 | 0.9 | User provided a better answer |
| Custom `score=0.7` | 0.7 | by type | Pass any float in [-1, 1] |

### Thumbs up / down

The simplest signal. Wire to a button in your UI:

```python
client.feedback(response.id, "thumbs_up")
client.feedback(response.id, "thumbs_down", comment="Too verbose")
```

### Corrections

The most valuable signal — the user tells you exactly what they wanted:

```python
client.feedback(
    response.id,
    "correction",
    comment="The correct answer is: boil water for 10 minutes, not 5.",
)
```

The refiner sees the original response AND the correction, learning the gap.

### Custom scores

For nuanced feedback (e.g., a 1-5 star rating):

```python
# Map stars to [-1, 1]: stars/5 * 2 - 1
stars = 4
score = stars / 5 * 2 - 1  # 0.6
client.feedback(response.id, "positive", score=score)
```

## Implicit signals (coming soon)

| Signal | Score | Confidence | Detection |
|--------|-------|-----------|-----------|
| `implicit_reask` | -0.4 | 0.5 | User re-asks the same question |
| `implicit_abandon` | -0.3 | 0.3 | User stops interacting |

You can record these manually today:

```python
client.feedback(interaction_id, "implicit_reask")
```

Automatic detection is planned for a future release.

## Feedback widget

For web apps, embed the HTML widget directly:

```python
html = client.get_widget_html(response.id, style="standard")
# Inject into your page template
```

Three styles: `minimal` (icon buttons), `standard` (buttons + comment box), `detailed` (buttons + tags + comment).

## Noise filtering

AutoRefine automatically filters noisy feedback:

- **Rage-clicking** — user spamming 5+ negatives in 2 minutes
- **Contradictions** — same user giving positive and negative on the same response
- **Outlier users** — one user responsible for >50% of negatives

Disable with `feedback_filter_enabled=False`.
