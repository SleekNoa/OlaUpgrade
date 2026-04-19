# agent/multi_agent.py
"""
Multi-Agent Orchestrator
========================
Routes user queries to the most appropriate specialized agent.

This gives us "least privilege" — each sub-agent only sees the tools it needs.
"""

import os
from tools.mcp_client import MCPClient
from agent.base_agent import BaseAgent

ORCHESTRATOR_PROMPT = """
You are a routing orchestrator. Given a user request, decide which specialist to invoke:
- "weather"     → for any weather-related questions
- "allocation"  → for task or resource allocation questions
- "general"     → for everything else

Reply with ONLY one word: weather | allocation | general
""".strip()

WEATHER_PROMPT = "You are a weather assistant. Use tools to answer weather questions clearly."
ALLOC_PROMPT   = "You are a resource planner. Use tools to allocate tasks to agents optimally."
GENERAL_PROMPT = "You are a helpful assistant. Answer the user's question."


class MultiAgentOrchestrator:
    """
    Two-phase execution:
    Phase 1 - Router agent classifies intent (no tool access).
    Phase 2 - Specialist agent handles request with relevant tools only.

    Adding a new specialist: add a prompt + entry in self.agents below.
    """

    def __init__(self, mcp: MCPClient, provider: str | None = None, model: str | None = None):
        provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        model = model or os.getenv("LLM_MODEL")

        kwargs = {"mcp": mcp, "provider": provider, "model": model}

        self.router = BaseAgent(
            "Router",
            system_prompt=ORCHESTRATOR_PROMPT,
            allowed_tools=[],
            **kwargs,
        )

        self.agents: dict[str, BaseAgent] = {
            "weather": BaseAgent(
                name="WeatherAgent",
                mcp=mcp,
                model=model,
                system_prompt=WEATHER_PROMPT,
                allowed_tools=["get_weather"],
            ),
            "allocation": BaseAgent(
                name="AllocationAgent",
                mcp=mcp,
                model=model,
                system_prompt=ALLOC_PROMPT,
                allowed_tools=["allocate_tasks"],
            ),
            "general": BaseAgent(
                name="GeneralAgent",
                mcp=mcp,
                model=model,
                system_prompt=GENERAL_PROMPT,
                allowed_tools=[],  # No tools
            ),
        }

    async def run(self, user_input: str) -> str:
        # Step 1: Decide which agent should handle this
        route_raw = await self.router.run(user_input)

        if not route_raw:
            print("Router returned empty response → defaulting to 'general'")
            route = "general"
        else:
            route_raw = route_raw.lower().strip()
            route = next((k for k in self.agents if k in route_raw), "general")

        print(f"  → Routed to: {route}")

        # Step 2: Let the chosen specialist handle it
        return await self.agents[route].run(user_input)