# llm/factory.py
import os
from llm.base import BaseLLM


def create_llm(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str = "",
) -> BaseLLM:
    """
    Creates the right LLM from env or explicit args.

    ENV vars:
      LLM_PROVIDER = ollama | openai | groq | lmstudio  (default: ollama)
      LLM_MODEL    = model name override
      OPENAI_API_KEY, OPENAI_BASE_URL (for OpenAI / compatible endpoints)

    Examples:
      create_llm()                            → Ollama llama3.1:8b
      create_llm("openai", "gpt-4o-mini")    → OpenAI GPT-4o-mini
      create_llm("openai", base_url=...)     → any OpenAI-compatible server
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    model    = model or os.getenv("LLM_MODEL")

    if provider == "ollama":
        from llm.ollama_client import OllamaLLM
        return OllamaLLM(
            model=model or "llama3.1:8b",
            system_prompt=system_prompt,
        )

    if provider in ("openai", "groq", "lmstudio", "compatible"):
        from llm.openai_client import OpenAILLM
        return OpenAILLM(
            model=model or "gpt-4o-mini",
            system_prompt=system_prompt,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),   # None = real OpenAI
        )

    raise ValueError(f"Unknown LLM provider: '{provider}'. Use ollama|openai|groq|lmstudio")