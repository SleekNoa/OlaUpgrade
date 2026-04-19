# agent/base_agent.py
"""
Base agent implementation for the tool-calling loop.

Loop shape:
1. Ask the LLM for the next step.
2. If it returns tool calls, execute them through MCP.
3. Inject the observations back into history.
4. Ask for the final plain-English answer.
"""

import json
from llm.base import BaseLLM
from llm.factory import create_llm
from tools.mcp_client import MCPClient


class BaseAgent:
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
        self.llm: BaseLLM = create_llm(provider, model, system_prompt)

    async def _get_tools(self) -> list[dict]:
        if self.allowed_tools == []:
            return []
        all_tools = await self.mcp.ollama_tool_schemas()
        if self.allowed_tools is None:
            return all_tools
        return [t for t in all_tools if t["function"]["name"] in self.allowed_tools]

    @staticmethod
    def _is_fake_tool_call(msg: dict) -> bool:
        """Catch models that emit tool JSON as plain text instead of real tool_calls."""
        content = msg.get("content") or ""
        return (
            '"type":"function"' in content.replace(" ", "") or
            "<|tool_call|>" in content or
            "<|/tool_call|>" in content
        )

    async def run(self, user_input: str) -> str:
        self.llm.reset()
        tools = await self._get_tools()
        msg = self.llm.chat(user_input, tools=tools or None)

        for _ in range(self.max_steps):
            if self._is_fake_tool_call(msg):
                print(f"  [{self.name}] Fake tool call detected, asking for plain answer")
                msg = self.llm.chat(
                    "Please answer the question directly in plain text. Do not output JSON or code.",
                    tools=None,
                )
                break

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                break

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"].get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                print(f"  [{self.name}] {name}({args})")
                observation = await self.mcp.call_tool(name, args)
                print(f"  [{self.name}] {observation}")
                self.llm.inject_tool_result(name, observation)

            # After tools run, prompt for a natural-language answer instead of raw JSON.
            msg = self.llm.chat(
                "Based on the tool results above, give your final answer in plain English.",
                tools=tools or None,
            )

      # return (msg.get("content") or "").strip()
        final = (msg.get("content") or "").strip()

        print(f"  [{self.name}] FINAL MSG:", msg)
        print(f"  [{self.name}] FINAL TEXT:", final)

        return final
