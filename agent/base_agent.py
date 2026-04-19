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
from collections import deque

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
        max_steps: int = 8,
        allowed_tools: list[str] | None = None,
        max_repeated_tool_calls: int = 5,
    ):
        self.name = name
        self.mcp = mcp
        self.max_steps = max_steps
        self.allowed_tools = allowed_tools
        self.max_repeated_tool_calls = max_repeated_tool_calls

        self.llm: BaseLLM = create_llm(provider, model, system_prompt)

        # Loop detection
        self._recent_tool_signatures: deque[str] = deque(maxlen=20)

    async def _get_tools(self) -> list[dict]:
        if self.allowed_tools == []:
            return []

        all_tools = await self.mcp.ollama_tool_schemas()

        if self.allowed_tools is None:
            return all_tools

        return [t for t in all_tools if t["function"]["name"] in self.allowed_tools]

    @staticmethod
    def _is_fake_tool_call(msg: dict) -> bool:
        """Catch models that output tool JSON as text instead of proper tool_calls."""
        content = (msg.get("content") or "").replace(" ", "").lower()
        return (
            '"type":"function"' in content
            or "<|tool_call|>" in content
            or "<|/tool_call|>" in content
            or ('"function"' in content and '"name"' in content)
            or ('"name"' in content and '"parameters"' in content)
        )

    def _get_tool_signature(self, name: str, args: dict) -> str:
        """Canonical signature for loop detection (ignores arg order)."""
        try:
            return f"{name}:{json.dumps(args, sort_keys=True)}"
        except Exception:
            return f"{name}:{args}"

    async def run(self, user_input: str) -> str:
        self.llm.reset()
        self._recent_tool_signatures.clear()

        tools = await self._get_tools()

        # Initial LLM call with tools enabled
        msg = self.llm.chat(user_input, tools=tools or None)

        for _ in range(self.max_steps):

            if self._is_fake_tool_call(msg):
                print(f"  [{self.name}] Fake tool call detected → forcing plain answer")
                msg = self.llm.chat(
                    "Answer the original question directly in clear, natural English. "
                    "No JSON, no code blocks, no tool calls.",
                    tools=None,
                )
                break

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # LLM gave a direct final answer
                break

            broke_early = False

            # === Execute all tool calls in this turn ===
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                args = tc["function"].get("arguments", {})

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                signature = self._get_tool_signature(fn_name, args)
                self._recent_tool_signatures.append(signature)

                # Repeated tool call protection
                if list(self._recent_tool_signatures).count(signature) >= self.max_repeated_tool_calls:
                    print(f"  [{self.name}] Repeated tool call ({fn_name}). Breaking loop.")
                    msg = self.llm.chat(
                        "Summarize what you know and give a final answer in plain English.",
                        tools=None,
                    )
                    broke_early = True
                    break

                print(f"  [{self.name}] Calling {fn_name}({args})")
                observation = await self.mcp.call_tool(fn_name, args)

                obs_str = str(observation)
                print(f"  [{self.name}] Observation: {obs_str[:300]}{'...' if len(obs_str) > 300 else ''}")

                self.llm.inject_tool_result(fn_name, observation)

            if broke_early:
                break

            # After tools → ask for final answer (no tools allowed)
            if tool_calls:
                msg = self.llm.chat(
                    "Based on the tool results above, give your final answer in plain English. "
                    "Be concise and direct. No JSON, no code blocks.",
                    tools=None,
                )

        # ← Return is now correctly outside the loop
        return (msg.get("content") or "").strip()
