-- AutoRefine Cloud schema extension.
-- These tables extend the existing postgres_store.py schema.

-- Organizations
CREATE TABLE IF NOT EXISTS organizations (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT UNIQUE NOT NULL,
    plan            TEXT NOT NULL DEFAULT 'self_serve',
    markup_rate     DOUBLE PRECISION DEFAULT 0.35,
    monthly_spend_cap DOUBLE PRECISION DEFAULT 500.0,
    stripe_customer_id TEXT DEFAULT '',
    upstream_keys   JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE
);

-- API Keys (never store plaintext keys)
CREATE TABLE IF NOT EXISTS api_keys (
    id              TEXT PRIMARY KEY,
    org_id          TEXT NOT NULL REFERENCES organizations(id),
    key_hash        TEXT UNIQUE NOT NULL,
    key_prefix      TEXT NOT NULL,
    name            TEXT DEFAULT '',
    model_restrictions JSONB DEFAULT '[]',
    monthly_spend_cap DOUBLE PRECISION,
    rate_limit_rpm  INTEGER DEFAULT 600,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL,
    last_used_at    TIMESTAMPTZ
);

-- Usage records (the billing ledger)
CREATE TABLE IF NOT EXISTS usage_records (
    id              TEXT PRIMARY KEY,
    org_id          TEXT NOT NULL REFERENCES organizations(id),
    api_key_id      TEXT NOT NULL REFERENCES api_keys(id),
    interaction_id  TEXT DEFAULT '',
    model           TEXT NOT NULL,
    provider        TEXT NOT NULL,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    upstream_cost   DOUBLE PRECISION DEFAULT 0.0,
    markup_amount   DOUBLE PRECISION DEFAULT 0.0,
    customer_cost   DOUBLE PRECISION DEFAULT 0.0,
    prompt_key      TEXT DEFAULT 'default',
    created_at      TIMESTAMPTZ NOT NULL
);

-- Daily usage summaries (materialized by the daily sync job)
CREATE TABLE IF NOT EXISTS daily_usage_summaries (
    org_id          TEXT NOT NULL REFERENCES organizations(id),
    date            TEXT NOT NULL,
    total_calls     INTEGER DEFAULT 0,
    total_input_tokens  INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_upstream_cost DOUBLE PRECISION DEFAULT 0.0,
    total_markup    DOUBLE PRECISION DEFAULT 0.0,
    total_customer_cost DOUBLE PRECISION DEFAULT 0.0,
    models_used     JSONB DEFAULT '{}',
    PRIMARY KEY (org_id, date)
);

-- Indexes for billing queries
CREATE INDEX IF NOT EXISTS idx_usage_org_created ON usage_records(org_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_key_created ON usage_records(api_key_id, created_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_org ON api_keys(org_id);
CREATE INDEX IF NOT EXISTS idx_orgs_slug ON organizations(slug);
