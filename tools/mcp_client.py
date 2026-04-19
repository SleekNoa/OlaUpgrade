# tools/mcp_client.py
"""
MCP client bridge used by the agents.

Responsibilities:
- Launch the MCP server subprocess
- Discover tool schemas for the LLM
- Execute tool calls and normalize responses
"""

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, server_script: str = "mcp-server/mcp_server.py"):
        self.server_script = server_script
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._schema_cache: list[dict] | None = None

    async def __aenter__(self):
        """Start the MCP server subprocess and open a client session."""
        project_root = str(Path(self.server_script).parent.parent.resolve())
        print(f"Launching MCP server from project root: {project_root}")

        # PYTHONPATH makes sibling packages importable when the subprocess starts.
        env = {**os.environ, "PYTHONPATH": project_root}

        params = StdioServerParameters(
            command="python",
            args=[self.server_script],
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
        """Return MCP tools in the function schema shape expected by the LLMs."""
        if self._schema_cache is not None:
            return self._schema_cache

        tools = (await self.session.list_tools()).tools
        self._schema_cache = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]
        return self._schema_cache

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Execute one MCP tool call and coerce the result into Python data."""
        if not self.session:
            raise RuntimeError("MCPClient not connected.")

        result = await self.session.call_tool(name, arguments)
        if result.content and hasattr(result.content[0], "text"):
            try:
                return json.loads(result.content[0].text)
            except json.JSONDecodeError:
                return {"result": result.content[0].text}
        return {"result": str(result)}
