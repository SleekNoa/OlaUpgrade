# api/server.py
"""
OpenAI-compatible HTTP wrapper around the multi-agent system.

The FastAPI app keeps one shared MCP client + orchestrator alive for the
process lifetime so we do not restart the MCP subprocess on every request.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from tools.mcp_client import MCPClient
from agent.multi_agent import MultiAgentOrchestrator

_mcp: MCPClient | None = None
_orchestrator: MultiAgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start shared dependencies once at boot and close them on shutdown."""
    global _mcp, _orchestrator
    print("Starting MCP client and orchestrator...")
    _mcp = MCPClient()
    await _mcp.__aenter__()
    _orchestrator = MultiAgentOrchestrator(_mcp)
    print("OLA Agent system is ready - MCP connected successfully")
    yield
    if _mcp:
        await _mcp.__aexit__(None, None, None)
    print("Shutdown complete")


app = FastAPI(title="OLA Agent API", lifespan=lifespan)


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
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ola-agent"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "ola-agent", "object": "model", "owned_by": "local"}],
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><head><title>OLA Agent</title></head>
    <body style="font-family:Arial;max-width:700px;margin:40px auto">
        <h1>OLA Agent</h1>
        <p>Status: <strong>Running</strong></p>
        <p>Endpoint: <code>POST http://localhost:8000/v1/chat/completions</code></p>
        <p><a href="/health">Health</a> | <a href="/v1/models">Models</a></p>
    </body></html>
    """


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """Accept the last user message and return an OpenAI-style completion."""
    user_msgs = [m for m in req.messages if m.role.lower() == "user"]
    if not user_msgs:
        return JSONResponse(status_code=400, content={"error": "No user message found"})

    user_input = user_msgs[-1].content.strip()
    print(f"Received: {user_input[:120]}")

    answer = await _orchestrator.run(user_input)
    print(f"Agent response generated ({len(answer)} characters)")

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=req.model,
        choices=[Choice(
            index=0,
            message=Message(role="assistant", content=answer),
            finish_reason="stop",
        )],
    )
