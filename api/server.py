# api/server.py
"""
OLA Agent API Server - OpenAI Compatible Gateway
================================================

This file turns your multi-agent orchestration system into a standard
OpenAI-compatible HTTP API.

Why this design?
- Most chat UIs (Open WebUI, OlaChat, SillyTavern, LM Studio, LangChain, etc.)
  expect the `/v1/chat/completions` endpoint format.
- This lets you connect your custom agents to any existing frontend with zero frontend work.
- The MCP server and orchestrator are started once at boot (efficient) and reused across requests.

How to use:
- Base URL: http://localhost:8000
- Model name: anything (we ignore it for now)

Supported clients: Open WebUI, curl, Postman, LangChain, etc.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Core imports
from tools.mcp_client import MCPClient
from agent.multi_agent import MultiAgentOrchestrator

# ── Shared Application State ─────────────────────────────────────────────
# These are initialized once when the server starts and reused for all requests.
# This avoids restarting the MCP server (and its tools) on every chat.
_mcp: MCPClient | None = None
_orchestrator: MultiAgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - runs on server startup and shutdown.

    What it does:
    1. Creates and starts the MCPClient (launches mcp_server.py subprocess)
    2. Creates the MultiAgentOrchestrator with access to the shared MCP client
    3. Yields control so FastAPI can start accepting requests
    4. On shutdown: cleanly closes the MCP connection

    This prevents resource leaks and "zombie" MCP processes.
    """
    global _mcp, _orchestrator

    print("🔄 Starting MCP client and orchestrator...")

    _mcp = MCPClient()
    await _mcp.__aenter__()  # Starts the MCP server process

    _orchestrator = MultiAgentOrchestrator(_mcp)

    print("🚀 OLA Agent system is ready - MCP connected successfully")

    yield  # ← Server now runs and handles requests

    # Cleanup on shutdown
    if _mcp:
        await _mcp.__aexit__(None, None, None)
    print("👋 OLA Agent server shutting down")


# Create the FastAPI application with lifespan handler
app = FastAPI(
    title="OLA Agent API",
    description="OpenAI-compatible endpoint for multi-agent orchestration with MCP tools",
    lifespan=lifespan
)


# ── Pydantic Models (Match OpenAI Chat Completions API) ───────────────────
class Message(BaseModel):
    """Standard message format used by OpenAI-compatible clients."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """
    Incoming request from OpenAI clients.

    We mainly use:
    - messages: full conversation history (we only take the last user message)
    - model: ignored for now (can be used later for model switching)
    """
    model: str = "ola-agent"
    messages: list[Message]
    stream: bool = False  # Streaming not implemented yet


class Choice(BaseModel):
    """One completion choice (OpenAI format)."""
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    """Full response matching the official OpenAI /v1/chat/completions format."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]


# ── API Routes ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple health check endpoint - useful for Docker, monitoring, etc."""
    return {"status": "ok", "service": "ola-agent-api"}


@app.get("/v1/models")
async def list_models():
    """
    Required by many OpenAI-compatible frontends (Open WebUI, etc.).
    Returns a fake model so the UI shows "ola-agent" as available.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "ola-agent",
                "object": "model",
                "owned_by": "local",
                "created": int(time.time())
            }
        ]
    }


@app.get("/", response_class=HTMLResponse)
async def root_page():
    """
    Simple browser homepage with basic test form.
    Makes it easy to test the agent directly in the browser.
    """
    return """
    <html>
    <head><title>OLA Agent</title>
    <style>body { font-family: Arial, sans-serif; margin: 40px; max-width: 800px; }</style>
    </head>
    <body>
        <h1>🚀 OLA Agent Server</h1>
        <p><strong>Status:</strong> Running • MCP Tools Loaded</p>

        <h2>Quick Test</h2>
        <form action="/test" method="post">
            <textarea name="message" rows="4" cols="80" placeholder="Ask anything... e.g. What's the weather in Marion, IA?"></textarea><br><br>
            <button type="submit">Send to Agent</button>
        </form>

        <p>
            <a href="/health">Health Check</a> | 
            <a href="/v1/models">List Models</a>
        </p>
    </body>
    </html>
    """


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> ChatResponse:
    """
    Main OpenAI-compatible chat endpoint.

    Logic flow:
    1. Extract the LAST user message from the conversation history
       (clients usually send the full history)
    2. Pass that input to the MultiAgentOrchestrator
       → Orchestrator routes to the right specialist agent
       → Agent runs the CCA tool-calling loop (LLM → tools → observations → final answer)
    3. Wrap the final answer in standard OpenAI response format

    Note: Streaming is not yet implemented (can be added later with Server-Sent Events).
    """
    # Extract the most recent user message
    user_messages = [m for m in req.messages if m.role.lower() == "user"]
    if not user_messages:
        return JSONResponse(
            status_code=400,
            content={"error": "No user message found in request"}
        )

    user_input = user_messages[-1].content.strip()

    if not user_input:
        return JSONResponse(status_code=400, content={"error": "User message cannot be empty"})

    print(f"📥 Received: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")

    # Run the full multi-agent system
    answer = await _orchestrator.run(user_input)

    print(f"📤 Agent response generated ({len(answer)} characters)")

    # Return in exact OpenAI format
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
    )


# # Optional: Simple test endpoint for browser form
# @app.post("/test")
# async def quick_test(message: str):
#     """Simple endpoint for the homepage test form."""
#     if not _orchestrator:
#         return {"error": "Orchestrator not initialized"}
#
#     answer = await _orchestrator.run(message)
#     return {"user_input": message, "agent_response": answer}
