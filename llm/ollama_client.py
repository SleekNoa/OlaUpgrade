# llm/ollama_client.py
"""
Ollama LLM Client - Stateful with Native Tool Calling
=====================================================
This is the core LLM interface for Ollama.

Important improvements over naive implementations:
- Maintains conversation history (critical for multi-turn + tool use)
- Uses native `tools=` parameter → Ollama returns `tool_calls` field
- No fragile text parsing like "TOOL_CALL:"
"""

from typing import Any
import ollama
from llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """
    Stateful wrapper around Ollama with proper tool calling support.
    """

    def __init__(self, model: str = "llama3.1:8b", system_prompt: str = ""):
        self.model = model
        self.system_prompt = system_prompt
        self.history: list[dict] = []   # Full conversation history

    def reset(self):
        """Clear history (start fresh conversation)."""
        self.history.clear()

    def chat(self, user_message: str, tools: list[dict] | None = None) -> dict:
        """
        Send a message to the LLM and get response.

        Returns the full assistant message dict, which may contain:
        - "content": normal text answer
        - "tool_calls": list of tools the model wants to call
        """
        # Add system prompt only once at the beginning
        if not self.history and self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

        self.history.append({"role": "system", "content": user_message})

        kwargs = dict[str, Any] = {"model": self.model, "messages": self.history}
        if tools:
            kwargs["tools"] = tools

        response = ollama.chat(**kwargs)    # dict with role/content/tool_calls
        assistant_msg = response["message"]

        # Save assistant response to history for context in next turn
        self.history.append(assistant_msg)  # keep in context
        return assistant_msg

    def inject_tool_result(self, tool_name: str, result: Any) -> None:
        """
        Add the result of a tool call back into the conversation history.
        This is the "observation" step in the tool-calling loop.
        """
        self.history.append({
            "role": "tool",
            "name": tool_name,
            "content": str(result),
        })


