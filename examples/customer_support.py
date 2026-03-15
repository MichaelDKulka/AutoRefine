"""Multi-category customer support with independent prompt refinement.

Each support category (billing, technical) has its own prompt_key.
Feedback on billing responses only improves the billing prompt.

Run:
    pip install autorefine[openai]
    python examples/customer_support.py
"""

from autorefine import AutoRefine

# -- Create separate clients per category ----------------------------------
# Each prompt_key evolves independently based on category-specific feedback.

billing_bot = AutoRefine(
    api_key="sk-your-openai-key",
    model="gpt-4o",
    refiner_key="sk-ant-your-claude-key",
    auto_learn=True,
    prompt_key="support-billing",  # independent prompt namespace
)
billing_bot.set_system_prompt(
    "You are a billing support agent. Help customers with invoices, "
    "payments, refunds, and subscription changes. Be empathetic and precise. "
    "Always reference the customer's account details when available."
)

technical_bot = AutoRefine(
    api_key="sk-your-openai-key",
    model="gpt-4o",
    refiner_key="sk-ant-your-claude-key",
    auto_learn=True,
    prompt_key="support-technical",  # separate namespace
)
technical_bot.set_system_prompt(
    "You are a technical support agent. Help customers troubleshoot issues "
    "with our software. Ask clarifying questions, suggest diagnostic steps, "
    "and provide clear solutions. Reference documentation when helpful."
)


# -- Route tickets to the right bot ---------------------------------------

def handle_ticket(category: str, message: str) -> tuple[str, str]:
    """Route a support ticket and return (response_id, text)."""
    bot = billing_bot if category == "billing" else technical_bot
    system = bot.get_active_prompt().system_prompt
    response = bot.complete(system, message)
    return response.id, response.text


def rate_response(category: str, response_id: str, rating: str, comment: str = ""):
    """Rate a support response -- feedback flows to the right category."""
    bot = billing_bot if category == "billing" else technical_bot
    bot.feedback(response_id, rating, comment=comment)


# -- Demo ------------------------------------------------------------------

if __name__ == "__main__":
    # Billing ticket
    rid, text = handle_ticket("billing", "I was charged twice for my subscription")
    print(f"Billing bot: {text}")
    rate_response("billing", rid, "thumbs_down", "Didn't offer a refund proactively")

    # Technical ticket
    rid, text = handle_ticket("technical", "The app crashes when I export a PDF")
    print(f"\nTech bot: {text}")
    rate_response("technical", rid, "thumbs_up")

    # Check costs per category
    print(f"\nBilling costs: {billing_bot.costs}")
    print(f"Tech costs: {technical_bot.costs}")
