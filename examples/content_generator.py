"""Content writing assistant that improves from editorial feedback.

Uses 'correction' feedback to teach the model your editorial voice.
Each time an editor rewrites a response, the refiner learns the gap.

Run:
    pip install autorefine[openai]
    python examples/content_generator.py
"""

from autorefine import AutoRefine

# -- Setup -----------------------------------------------------------------
SYSTEM = (
    "You are a professional content writer. Write engaging, clear, and "
    "well-structured content. Match the tone to the audience. Use active "
    "voice. Keep paragraphs short."
)

client = AutoRefine(
    api_key="sk-your-openai-key",
    model="gpt-4o",
    refiner_key="sk-ant-your-claude-key",
    auto_learn=True,
    prompt_key="content-writer",  # dedicated namespace
)
client.set_system_prompt(SYSTEM)


# -- Generate content ------------------------------------------------------

def generate(topic: str, audience: str = "general") -> tuple[str, str]:
    """Generate an article and return (response_id, text)."""
    prompt = f"Write a short article about: {topic}\nTarget audience: {audience}"
    response = client.complete(SYSTEM, prompt)
    return response.id, response.text


def submit_editorial(response_id: str, edited_version: str):
    """Submit an editor's rewrite as correction feedback.

    The refiner will learn from the delta between the original and
    the editor's version to improve tone, structure, and accuracy.
    """
    client.feedback(response_id, "correction", comment=edited_version)


# -- Demo ------------------------------------------------------------------

if __name__ == "__main__":
    # Generate an article
    rid, article = generate("The benefits of remote work", audience="tech professionals")
    print("Generated article:")
    print(article)
    print("\n---\n")

    # Simulate an editor improving the output
    edited = (
        "Remote work has transformed how tech teams operate. Here are three "
        "concrete benefits backed by data:\n\n"
        "1. **Productivity gains** -- Studies show a 13% performance increase.\n"
        "2. **Talent access** -- Companies can hire from a global pool.\n"
        "3. **Cost savings** -- Both employees and employers save on overhead.\n\n"
        "The key is intentional communication and clear async workflows."
    )
    submit_editorial(rid, edited)
    print("Editorial feedback submitted -- future articles will improve!\n")

    # Check analytics
    snap = client.get_analytics(days=1)
    print(f"Total feedback: {snap.total_feedback}")
    print(f"Active prompt version: v{snap.active_version}")
