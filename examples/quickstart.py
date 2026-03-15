"""AutoRefine Quickstart -- your AI gets smarter in 10 lines.

Run:
    pip install autorefine[openai]
    export AUTOREFINE_API_KEY=sk-...
    python examples/quickstart.py
"""

from autorefine import AutoRefine

# 1. Create a client (model auto-detects provider from name)
client = AutoRefine(
    api_key="sk-your-openai-key",       # or set AUTOREFINE_API_KEY env var
    model="gpt-4o",                      # works with any OpenAI/Anthropic/Mistral/Ollama model
    refiner_key="sk-ant-your-key",       # optional: enables automatic prompt refinement
    auto_learn=True,                     # refine after enough feedback accumulates
)

# 2. Set your initial system prompt
client.set_system_prompt("You are a helpful cooking assistant.")

# 3. Use it like any LLM client
response = client.complete("You are a helpful cooking assistant.", "How do I make pasta carbonara?")
print(response.text)

# 4. Collect feedback -- wire this to your app's thumbs up/down buttons
client.feedback(response.id, "thumbs_up")

# That's it! After enough feedback, your prompt auto-refines.
# Check the current version:
active = client.get_active_prompt()
if active:
    print(f"\nActive prompt v{active.version}: {active.system_prompt[:80]}...")
