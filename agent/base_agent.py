# agent/base_agent.py
"""
BaseAgent - Implements the Tool Calling Loop
======================================================================
This is the heart of the agent.

It follows the standard modern tool-calling pattern:
1. Send user message + tool schemas to LLM
2. If LLM returns tool_calls → execute them via MCP
3. Inject results back as "tool" role messages
4. Repeat until LLM gives a final text answer
"""

import json
from llm.base import BaseLLM
from llm.factory import create_llm
from tools.mcp_client import MCPClient

class BaseAgent:
    """
    CCA-style single agent:
    1. Fetches Ollama tool schemas from MCP.
    2. Calls LLM with schemas injected.
    3. If LLM emits tool_calls → dispatch via MCP → inject observation.
    4. Loop until text answer or max_steps reached.
    """

    def __init__(
            self,
            name: str,
            mcp: MCPClient,
            system_prompt: str = "",
            provider: str | None = None,
            model: str | None = None,
            max_steps: int = 6,
            allowed_tools: list[str] | None = None,
    ):
        self.name = name
        self.mcp = mcp
        self.max_steps = max_steps
        self.allowed_tools = allowed_tools
        # Each agent owns its own LLM instance (own history, own system prompt)
        self.llm: BaseLLM = create_llm(provider, model, system_prompt)

    async def _get_tools(self) -> list[dict]:
        all_tools = await self.mcp.ollama_tool_schemas()
        if self.allowed_tools is None:
            return all_tools
        return [t for t in all_tools if t["function"]["name"] in self.allowed_tools]

    def _parse_args(self, raw: dict | str) -> dict:
        """Normalise tool arguments - OpenAI returns JSON string, Ollama returns dict."""
        if isinstance(raw, str):
            return json.loads(raw)
        return raw or {}

    async def run(self, user_input: str) -> str:
        """Run the full agent loop until we get a final answer."""
        self.llm.reset()
        tools = await self._get_tools()

        msg = self.llm.chat(user_input, tools=tools)

        for step in range(self.max_steps):
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                break           # LLM gave a text answer / final answer

            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                args = fn.get("arguments", {})

                # Some models return arguments as JSON string
                if isinstance(args, str):
                    args = json.loads(args)

                print(f" [{self.name} <-> {name}({args})")

                observation = await self.mcp.call_tool(name, args)
                print(f"  [{self.name}] <0> {observation}")

                self.llm.inject_tool_result(name, observation)

                # Let the LLM reason over the new observations
                msg = self.llm.chat(
                    "Based on the tool results above, continue reasoning or give your final answer.",
                tools=tools,
                )

            return msg.get("content", "").strip()

