# llm/openai_client.py
from typing import Any
from openai import OpenAI
from llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI adapter — same interface as OllamaLLM.
    Also works with any OpenAI-compatible endpoint
    (Groq, Together, local vLLM, LM Studio, etc.)
    by setting base_url in .env.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = "",
        api_key: str | None = None,
        base_url: str | None = None,   # override for compatible endpoints
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.history: list[dict] = []
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,          # None = default OpenAI
        )

    def reset(self) -> None:
        self.history.clear()

    def chat(self, user_message: str, tools: list[dict] | None = None) -> dict:
        if not self.history and self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

        self.history.append({"role": "user", "content": user_message})

        kwargs: dict[str, Any] = {"model": self.model, "messages": self.history}
        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = "auto"

        response  = self.client.chat.completions.create(**kwargs)
        raw       = response.choices[0].message

        # Normalise to the same dict shape Ollama uses
        assistant: dict[str, Any] = {
            "role":    "assistant",
            "content": raw.content or "",
        }
        if raw.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,   # JSON string from OpenAI
                    },
                }
                for tc in raw.tool_calls
            ]
        self.history.append(assistant)
        return assistant

    def inject_tool_result(self, tool_name: str, result: Any) -> None:
        self.history.append({
            "role":    "tool",
            "name":    tool_name,
            "content": str(result),
        })