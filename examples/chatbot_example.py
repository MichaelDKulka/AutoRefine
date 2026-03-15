"""Interactive terminal chatbot that gets smarter from your feedback.

Each response gets a rating. After enough feedback, the system prompt
auto-refines and you'll see the bot's behavior improve.

Run:
    pip install autorefine[openai]
    python examples/chatbot_example.py
"""

from autorefine import AutoRefine

# -- Config ----------------------------------------------------------------
SYSTEM = "You are a friendly, concise assistant. Keep answers under 3 sentences."

client = AutoRefine(
    api_key="sk-your-openai-key",
    model="gpt-4o",
    refiner_key="sk-ant-your-claude-key",
    auto_learn=True,
    refine_threshold=5,  # low threshold for demo (default: 20)
)
client.set_system_prompt(SYSTEM)

# -- Chat loop -------------------------------------------------------------
print("Chat with the bot! Type 'quit' to exit.")
print("After each response, rate it: good / bad / Enter to skip.\n")

conversation: list[dict] = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break

    # Add user message to conversation history
    conversation.append({"role": "user", "content": user_input})

    # Send to LLM via AutoRefine (prompt is swapped in automatically)
    response = client.chat(SYSTEM, conversation)
    print(f"Bot: {response.text}\n")

    # Keep conversation history for multi-turn
    conversation.append({"role": "assistant", "content": response.text})

    # Collect feedback
    rating = input("Rate (good/bad/Enter to skip): ").strip().lower()
    if rating == "good":
        client.feedback(response.id, "thumbs_up")
        print("  -> Recorded positive feedback\n")
    elif rating == "bad":
        comment = input("  What was wrong? ").strip()
        client.feedback(response.id, "thumbs_down", comment=comment)
        print("  -> Recorded negative feedback\n")

# -- Session summary -------------------------------------------------------
snap = client.get_analytics(days=1)
print(f"\nSession stats:")
print(f"  Interactions: {snap.total_interactions}")
print(f"  Feedback: {snap.total_feedback}")
print(f"  Avg score: {snap.average_score:.2f}")
print(f"  Active prompt version: v{snap.active_version}")

active = client.get_active_prompt()
if active and active.version > 1:
    print(f"\n  Your prompt was refined! New version:\n  {active.system_prompt}")
