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

from dotenv import dotenv_values
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(
        self,
        server_script: str = "mcp-server/mcp_server.py",
        tool_name_prefixes: list[str] | None = None
    ):
        self.server_script = server_script
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._schema_cache: list[dict] | None = None

        self.tool_name_prefixes = tool_name_prefixes or [
            "ola-tools-",
            "anythingllm-tool-",
            "tool-"
        ]

    async def __aenter__(self):
        """Start the MCP server subprocess and open a client session."""
        project_root = str(Path(self.server_script).parent.parent.resolve())
        print(f"Launching MCP server from project root: {project_root}")

        # Load .env
        env_file = Path(project_root) / ".env"
        dotenv_vars = dotenv_values(env_file) if env_file.exists() else {}

        # === DEBUG INFO ===
        print(f".env file exists: {env_file.exists()}")
        print(f"Loaded dotenv keys: {list(dotenv_vars.keys())}")
        print(
            f"NEWS_API_KEY present in dotenv: "
            f"{'NEWS_API_KEY' in dotenv_vars and bool(dotenv_vars.get('NEWS_API_KEY'))}"
        )

        # Merge environments
        env = {
            **os.environ,
            **dotenv_vars,
            "PYTHONPATH": project_root,
            "MCP_DEBUG": "1",
        }

        # Extra safety: explicitly pass common keys
        for key in ["NEWS_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "WEATHER_API_KEY"]:
            if key in os.environ and key not in env:
                env[key] = os.environ[key]

        # 🔥 HARD DEBUG (important)
        print("\nFINAL ENV KEYS PASSED TO MCP:")
        print({k: ("SET" if v else "EMPTY") for k, v in env.items() if "KEY" in k})
        print()

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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute one MCP tool call with name normalization for external agents."""
        if not self.session:
            raise RuntimeError("MCPClient not connected.")

        original_name = name

        # Normalize tool name
        for prefix in self.tool_name_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        print(f"[MCP DEBUG] Calling tool: {original_name} → {name}")
        print(f"[MCP DEBUG] Arguments: {arguments}")

        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as e:
            return {
                "is_error": True,
                "error": {
                    "code": "TOOL_EXECUTION_FAILED",
                    "message": str(e),
                },
                "tool_name": name,
            }

        if not result.content:
            return {"result": None}

        text = (
            result.content[0].text
            if hasattr(result.content[0], "text")
            else str(result.content[0])
        )

        # Try parsing JSON
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"result": text}

        # 🔥 Normalize tool errors
        if isinstance(parsed, dict) and parsed.get("error"):
            parsed["is_error"] = True
            parsed["tool_name"] = name

        return parsed