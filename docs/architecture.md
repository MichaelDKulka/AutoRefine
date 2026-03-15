# Architecture

## The learning loop

```mermaid
sequenceDiagram
    participant App as Your App
    participant IC as Interceptor
    participant LLM as LLM Provider
    participant Store as Store
    participant FB as Feedback Collector
    participant Ref as Refiner (Claude)
    participant AB as A/B Tester

    App->>IC: chat(system, messages)
    IC->>Store: get_active_prompt()
    Store-->>IC: refined prompt v3
    IC->>LLM: chat(refined_prompt, messages)
    LLM-->>IC: response
    IC->>Store: save_interaction()
    IC-->>App: CompletionResponse(id, text)

    App->>FB: feedback(id, "thumbs_down", "too verbose")
    FB->>Store: save_feedback()

    Note over FB,Ref: After 20 signals...

    FB->>Ref: trigger refinement
    Ref->>Store: get feedback + interactions
    Ref->>LLM: meta-prompt with evidence
    LLM-->>Ref: {"new_prompt": "...", "changelog": [...]}
    Ref->>AB: start_test(candidate)

    Note over AB: A/B test runs (20% candidate traffic)

    AB->>Store: record results
    AB->>AB: Welch's t-test
    AB->>Store: promote winner → v4
```

## Module map

```
autorefine/
├── client.py            # Public API (AutoRefine class)
├── async_client.py      # Async version (AsyncAutoRefine)
├── interceptor.py       # Invisible middleware
├── feedback.py          # Feedback ingestion + batching
├── refiner.py           # Meta-prompt + Claude refinement
├── ab_testing.py        # Welch's t-test A/B validation
├── models.py            # Pydantic data models
├── config.py            # Pydantic settings (env vars)
├── _retry.py            # Exponential backoff retry
├── exceptions.py        # Exception hierarchy
├── providers/
│   ├── base.py          # Abstract base (sync + async)
│   ├── openai_provider  # OpenAI + compatible APIs
│   ├── anthropic_provider # Anthropic Claude
│   ├── ollama_provider  # Ollama (local models)
│   └── mistral_provider # Mistral AI
├── storage/
│   ├── base.py          # Abstract store interface
│   ├── json_store.py    # JSON file (dev)
│   ├── sqlite_store.py  # SQLite (production)
│   └── postgres_store.py # PostgreSQL (distributed)
├── pii_scrubber.py      # Regex PII redaction
├── feedback_filter.py   # Noise filtering
├── notifications.py     # Webhook + callback alerts
├── cost_tracker.py      # Cost tracking + budget
├── analytics.py         # Metrics + ROI reports
├── widget.py            # Embeddable HTML widget
├── cli.py               # CLI (click)
└── dashboard/
    ├── server.py         # FastAPI app
    ├── api.py            # REST endpoints
    ├── widget_endpoint.py # Widget feedback handler
    └── templates/
        └── index.html    # Dashboard SPA
```

## Data flow

```mermaid
graph TB
    subgraph Developer
        APP[Application Code]
    end

    subgraph AutoRefine SDK
        INT[Interceptor]
        FC[Feedback Collector]
        REF[Refiner]
        ABT[A/B Tester]
        PII[PII Scrubber]
        FF[Feedback Filter]
    end

    subgraph External
        LLM[LLM Provider]
        CLAUDE[Refiner Model]
    end

    subgraph Storage
        ST[(Store)]
    end

    APP --> INT
    INT --> LLM
    LLM --> INT
    INT --> ST
    APP --> FC
    FC --> FF
    FF --> ST
    ST --> REF
    REF --> PII
    PII --> CLAUDE
    CLAUDE --> REF
    REF --> ABT
    ABT --> ST
```

## Design principles

1. **Invisible** — The interceptor never raises exceptions from AutoRefine internals. Provider errors propagate; everything else is logged and swallowed.

2. **Surgical** — The refiner patches gaps rather than rewriting from scratch. Conditional logic over absolute rules.

3. **Conservative** — A/B testing with statistical significance. No candidate is promoted without evidence.

4. **Privacy-first** — PII scrubbed before reaching the refiner. Feedback noise filtered before refinement.

5. **Budget-aware** — Monthly cost caps with automatic refinement pausing.
