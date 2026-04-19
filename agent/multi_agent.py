# agent/multi_agent.py
"""Simple router + specialist orchestrator for the OLA agent."""

import os
from tools.mcp_client import MCPClient
from agent.base_agent import BaseAgent

ORCHESTRATOR_PROMPT = """
You are a routing agent. Classify the user request into exactly one word:
  weather    -> questions about weather or temperature
  allocation -> task assignment, resource planning, scheduling
  general    -> everything else

Reply with ONLY that one word. No punctuation, no explanation.
""".strip()

WEATHER_PROMPT = """
You are a weather assistant with access to the get_weather tool.

Rules:
- ALWAYS call get_weather to get real data. Never guess temperatures.
- After receiving tool results, respond in plain English only.
- State temperature in both Celsius and Fahrenheit.
- Do NOT output JSON, code, or tool syntax in your final answer.
""".strip()

ALLOC_PROMPT = """
You are a resource allocation planner with access to the allocate_tasks tool.

Rules:
- ALWAYS call allocate_tasks with the agents and tasks provided.
- After receiving results, explain each assignment in plain English.
- Do NOT output JSON or code in your final answer.
""".strip()

GENERAL_PROMPT = "You are a helpful assistant. Answer clearly and concisely in plain English."


class MultiAgentOrchestrator:
    def __init__(self, mcp: MCPClient, provider: str | None = None, model: str | None = None):
        provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        model = model or os.getenv("LLM_MODEL")
        kwargs = {"mcp": mcp, "provider": provider, "model": model}

        # Router has no tool access; it only decides which specialist should run.
        self.router = BaseAgent(
            "Router",
            system_prompt=ORCHESTRATOR_PROMPT,
            allowed_tools=[],
            **kwargs,
        )
        self.agents: dict[str, BaseAgent] = {
            "weather": BaseAgent("WeatherAgent", system_prompt=WEATHER_PROMPT,
                                  allowed_tools=["get_weather"], **kwargs),
            "allocation": BaseAgent("AllocationAgent", system_prompt=ALLOC_PROMPT,
                                     allowed_tools=["allocate_tasks"], **kwargs),
            "general": BaseAgent("GeneralAgent", system_prompt=GENERAL_PROMPT,
                                  allowed_tools=[], **kwargs),
        }

    async def run(self, user_input: str) -> str:
        route_raw = (await self.router.run(user_input)).lower().strip()
        route = next((k for k in self.agents if k in route_raw), "general")
        print(f"  -> Routed to: {route}")
        return await self.agents[route].run(user_input)
