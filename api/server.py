# api/server.py
"""
OLA Agent API Server - OpenAI Compatible Gateway
================================================

This file exposes your multi-agent system as a standard OpenAI-compatible API.

Why this design?
- Many frontends (Open WebUI, OlaChat, SillyTavern, LangChain, etc.) expect the
  `/v1/chat/completions` endpoint format.
- This allows you to plug your custom agent system into any existing chat UI
  without writing a custom frontend.
- The MCP server and orchestrator run once at startup (efficient) and are reused.

Point any OpenAI-compatible client here:
  base_url = http://localhost:8000
  model    = (anything — we ignore it and use LLM_MODEL from env)

Works with: OlaChat, Open WebUI, curl, Postman, LangChain, etc.

Architecture Flow:
User → OpenAI client (Open WebUI)
    → POST /v1/chat/completions
        → Extract last user message
            → MultiAgentOrchestrator.run()
                → Router → Specialist Agent → Tool calls via MCP
                    → Final answer returned as assistant message
"""''

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tools.mcp_client import MCPClient
from agent.multi_agent import MultiAgentOrchestrator

# Shared state
_mcp: MCPClient | None = None
_orchestrator: MultiAgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start MCP connection once at server boot, close on shutdown."""
    global _mcp, _orchestrator
    _mcp = MCPClient()
    await _mcp.__aenter__()
    _orchestrator = MultiAgentOrchestrator(_mcp)
    print("Lets Go, Agent server ready")
    yield
    await _mcp.__aexit__(None, None, None)
    print("Goodbye, Agent server shut down")


app = FastAPI(title="OLA Agent API", lifespan=lifespan)

# Request / Response models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "ola-agent"
    messages: list[Message]
    stream: bool = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]

# Routes
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """Required by some clients (Open WebUI, etc.)"""
    return {
        "object": "list",
        "data": [{"id": "ola-agent", "object": "model", "owned_by": "local"}],
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> ChatResponse:
    """
    OpenAI-compatible endpoint.
    Takes the last user message and runs it through the multi-agent orchestrator.
    """
    # Extract the last user message
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        return JSONResponse(status_code=400, content={"error": "No user message found"})

    user_input = user_messages[-1].content

    answer = await _orchestrator.run(user_input)

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
