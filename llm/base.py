# llm/base.py
from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """
    Contract every LLM adapter must fulfill.
    Agents depend on this interface, never on a concrete client.
    """

    @abstractmethod
    def chat(self, user_message: str, tools: list[dict] | None = None) -> dict:
        """
        Send a message. Returns full message dict:
          {"role": "assistant", "content": "...", "tool_calls": [...]}
        tool_calls may be absent if LLM responds with text.
        """

    @abstractmethod
    def inject_tool_result(self, tool_name: str, result: Any) -> None:
        """Append a tool observation into conversation history."""

    @abstractmethod
    def reset(self) -> None:
        """Clear conversation history for a new session."""