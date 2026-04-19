# tools/mcp_client.py
r"""
MCP Client - Bridge between Agents and MCP Server
=================================================
This class handles communication with the MCP server (tools).

Responsibilities:
1. Start/stop the MCP server process
2. Convert MCP tools → Ollama/OpenAI compatible tool schemas
3. Execute tool calls and return results

This is the "shared tool layer" used by all agents.
"""

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    Manages a connection to the FastMCP server and provides:
    - Tool schema discovery (for LLM tool calling)
    - Safe tool execution
    """

    def __init__(self, server_script: str = "mcp-server/mcp_server.py"):
        self.server_script = server_script
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._tools_cache: list[dict] | None = None  # Ollama schema cache

    async def __aenter__(self):
        """Start the MCP server and initialize session."""
        # Resolve project root so providers/ is always findable
        project_root = str(Path(self.server_script).parent.parent.resolve())

        print(f"Launching MCP server from project root: {project_root}")

        # Build environment with PYTHONPATH pointing to project root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

        params = StdioServerParameters(
            command="python",
            args=[self.server_script],
            cwd=str(project_root),
            env=env,
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        stdio, write = stdio_transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()
        tools = (await self.session.list_tools()).tools
        print(f"MCP ready - tools: {[t.name for t in tools]}")
        return self

    async def __aexit__(self, *exc):
        await self._exit_stack.aclose()

    async def ollama_tool_schemas(self) -> list[dict]:
        """
        Returns tool schemas in the exact format Ollama (and OpenAI) expect.
        Cached after first call for performance.
        """
        if self._tools_cache is not None:
            return self._tools_cache

        resp = await self.session.list_tools()
        schemas = []

        for t in resp.tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            })

        self._tools_cache = schemas
        return  schemas

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Execute a tool via MCP and return the result."""
        if not self.session:
            raise RuntimeError("MCPClient not connected. Use 'async with MCPClient():'")

        result = await self.session.call_tool(name, arguments)

        # Try to parse JSON response if possible
        if result.content and hasattr(result.content[0], "text"):
            try:
                return json.loads(result.content[0].text)
            except json.JSONDecodeError:
                return {"result": result.content[0].text}

        return {"result": str(result)}



