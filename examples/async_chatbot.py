"""Async chatbot with AutoRefine — works with asyncio and FastAPI.

Shows two patterns:
1. Pure asyncio terminal chatbot
2. FastAPI web endpoint with streaming

Run the terminal version:
    python examples/async_chatbot.py

Run the FastAPI version:
    pip install fastapi uvicorn
    uvicorn examples.async_chatbot:app --reload
"""

from __future__ import annotations

import asyncio

from autorefine import AsyncAutoRefine

SYSTEM = "You are a concise, friendly assistant."

# ── Shared async client ──────────────────────────────────────────────

client = AsyncAutoRefine(
    api_key="sk-your-openai-key",
    model="gpt-4o",
    refiner_key="sk-ant-your-claude-key",
    auto_learn=True,
    refine_threshold=5,
)


# ── Pattern 1: Pure asyncio terminal chatbot ─────────────────────────

async def terminal_chat():
    """Interactive async chatbot in the terminal."""
    await client.set_system_prompt(SYSTEM)
    print("Async chatbot! Type 'quit' to exit.\n")

    conversation: list[dict] = []
    while True:
        user_input = await asyncio.to_thread(input, "You: ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        conversation.append({"role": "user", "content": user_input})
        resp = await client.chat(SYSTEM, conversation)
        print(f"Bot: {resp.text}\n")
        conversation.append({"role": "assistant", "content": resp.text})

        rating = await asyncio.to_thread(input, "Rate (good/bad/skip): ")
        if rating.strip().lower() == "good":
            await client.feedback(resp.id, "thumbs_up")
        elif rating.strip().lower() == "bad":
            comment = await asyncio.to_thread(input, "  What was wrong? ")
            await client.feedback(resp.id, "thumbs_down", comment=comment)

    snap = await client.get_analytics(days=1)
    print(f"\nFeedback: {snap.total_feedback} | Avg: {snap.average_score:.2f}")


# ── Pattern 2: FastAPI web endpoint ──────────────────────────────────

def create_app():
    """Create a FastAPI app with chat and feedback endpoints."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        return None

    app = FastAPI(title="AutoRefine Async Chatbot")

    class ChatRequest(BaseModel):
        message: str
        system: str = SYSTEM

    class FeedbackRequest(BaseModel):
        response_id: str
        signal: str
        comment: str = ""

    @app.on_event("startup")
    async def startup():
        await client.set_system_prompt(SYSTEM)

    @app.post("/chat")
    async def chat(req: ChatRequest):
        resp = await client.complete(req.system, req.message)
        return {"id": resp.id, "text": resp.text, "model": resp.model}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        async def generate():
            async for chunk in await client.stream(
                req.system, [{"role": "user", "content": req.message}]
            ):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")

    @app.post("/feedback")
    async def feedback(req: FeedbackRequest):
        fb = await client.feedback(req.response_id, req.signal, comment=req.comment)
        return {"id": fb.id, "score": fb.score}

    @app.get("/health")
    async def health():
        return await client.health_check()

    return app


# Try to create the FastAPI app (for `uvicorn examples.async_chatbot:app`)
app = create_app()


if __name__ == "__main__":
    asyncio.run(terminal_chat())
