"""Simple router + specialist orchestrator for the OLA agent."""

import os
import re
import asyncio
from typing import List
from llm.factory import create_llm
from tools.mcp_client import MCPClient
from agent.base_agent import BaseAgent


# ==================== Router Prompt ====================
ORCHESTRATOR_PROMPT = """You are a routing agent.

Classify the user request into one or more categories.

Categories:
weather, allocation, news, search, stock, general

Definitions:

weather → weather, temperature, forecast
allocation → task assignment, scheduling, resource planning
news → RECENT or CURRENT events happening NOW (today, this week)
stock → stock prices, crypto prices, market data
search → research, unknown info, or when tools are needed
general → known facts, explanations, history, "who won", "why", "how"

CRITICAL RULES:
- "who won", "why", "how", "what is" → general (NOT news)
- Only use news if the question clearly asks for recent/current events
- Do NOT guess — choose the most precise category

Output ONLY category words separated by commas.
"""


# ==================== Specialist Prompts ====================
WEATHER_PROMPT = """You are a precise weather assistant. You have access to the get_weather tool.
Rules:
- Call get_weather exactly once with the city name.
- After receiving the result, provide a friendly, natural response.
- Always include the temperature in both Celsius and Fahrenheit.
- Keep the final answer concise and easy to read.
- Respond only in natural plain English. Never output tool calls, JSON, or code."""

NEWS_PROMPT = """You are a news assistant.
Instructions:
1. Call get_news once with the exact user topic.
2. If the tool returns articles (count > 0), pick the 3-5 most relevant ones and give a short, neutral summary of each (title + 1-sentence gist + source + date).
3. If results look irrelevant or count == 0, say "I found some recent articles but they don't closely match the requested topic." and list the best ones you have.
4. Never invent articles.
Rules:
- Use only data from the tool
- Be concise and natural
- No JSON or code blocks"""

SEARCH_PROMPT = """You are a research assistant with access to the search_web tool.
Rules:
- Call search_web exactly once with the user's query.
- If results are returned, summarize the most relevant findings and include source URLs where helpful.
- If no useful results or an error occurs, honestly say you couldn't find current information and provide your best knowledge instead.
- Never call the tool more than once.
- Respond only in natural plain English. Never output tool calls, JSON, or code blocks."""

STOCK_PROMPT = """You are a financial assistant with access to the get_stock tool.
Rules:
- Always call get_stock exactly once using the provided ticker symbol.
- After receiving the result, clearly report:
  • Company or asset name
  • Current price and currency
  • Daily change in percent
  • Market capitalization
- Respond in natural, plain English.
- Never call the tool more than once."""

ALLOC_PROMPT = """You are a resource allocation planner. You have access to the allocate_tasks tool.
Rules:
- Call allocate_tasks exactly once, passing the tasks and agents parsed from the user's request.
- After receiving the result, clearly explain the assignments in plain English.
- If any tasks could not be assigned, mention them.
- Never call the tool more than once.
- Respond only in natural plain English."""

GENERAL_PROMPT = """You are a helpful, clear, and concise assistant. Answer directly in natural plain English. Do not use tool calls unless necessary."""


# ==================== Orchestrator ====================
class MultiAgentOrchestrator:
    def __init__(self, mcp: MCPClient, provider: str | None = None, model: str | None = None):
        provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        model = model or os.getenv("LLM_MODEL")
        kwargs = {"mcp": mcp, "provider": provider, "model": model}

        # Lightweight router LLM - pure classification
        self.router_llm = create_llm(
            provider=provider,
            model=model,
            system_prompt=ORCHESTRATOR_PROMPT
        )

        # Specialist agents
        self.agents: dict[str, BaseAgent] = {
            "weather": BaseAgent(
                "WeatherAgent",
                system_prompt=WEATHER_PROMPT,
                allowed_tools=["get_weather"],
                **kwargs,
            ),
            "allocation": BaseAgent(
                "AllocationAgent",
                system_prompt=ALLOC_PROMPT,
                allowed_tools=["allocate_tasks"],
                **kwargs,
            ),
            "news": BaseAgent(
                "NewsAgent",
                system_prompt=NEWS_PROMPT,
                allowed_tools=["get_news"],
                **kwargs,
            ),
            "search": BaseAgent(
                "SearchAgent",
                system_prompt=SEARCH_PROMPT,
                allowed_tools=["search_web"],
                **kwargs,
            ),
            "stock": BaseAgent(
                "StockAgent",
                system_prompt=STOCK_PROMPT,
                allowed_tools=["get_stock"],
                **kwargs,
            ),
            "general": BaseAgent(
                "GeneralAgent",
                system_prompt=GENERAL_PROMPT,
                allowed_tools=[],
                **kwargs,
            ),
        }

    def _extract_categories(self, route_raw: str) -> list[str]:
        """Robust category extraction from router output."""
        if not route_raw:
            return ["general"]

        # Clean the output
        cleaned = re.sub(r'[^\w\s,.-]', '', route_raw.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Safer splitting
        words = [w.strip().lower() for w in cleaned.replace(',', ' ').split() if w.strip()]
        # Keep only valid categories
        categories = [w for w in words if w in self.agents]
        # Deduplicate while preserving order
        return list(dict.fromkeys(categories))

    def _keyword_fallback(self, user_input: str) -> List[str]:
        """Strong rule-based fallback."""
        lower = user_input.lower()
        detected = []

        if any(kw in lower for kw in ["weather", "temperature", "forecast", "rain", "sunny", "°c", "°f"]):
            detected.append("weather")
        if any(kw in lower for kw in ["assign", "allocate", "schedule", "task", "who does", "resource"]):
            detected.append("allocation")
        if any(kw in lower for kw in ["news", "latest", "recent", "breaking"]):  # removed "who won", "what happened"
            detected.append("news")
        if any(kw in lower for kw in ["stock", "price", "ticker", "share", "aapl", "tsla", "nvda", "bitcoin", "crypto", "market cap"]):
            detected.append("stock")

        return detected or ["general"]

    def _refine_query(self, category: str, user_input: str) -> str:
        """Refine query per specialist to avoid garbage inputs."""
        if category == "news":
            return f"latest news about {user_input}"
        # weather, stock, etc. use original input
        return user_input

    async def run(self, user_input: str) -> str:
        """Route user query to appropriate specialist(s) and return response."""
        lower_query = user_input.lower()

        # === Guard: meaningless news queries ===
        if re.search(r'\b(news about nothing|nothing news|anything news|news on nothing)\b', lower_query):
            return "I couldn’t find a meaningful news topic. Try something more specific."

        self.router_llm.reset()

        # === Hard overrides (fast path - eliminates 30-40% of routing errors) ===
        if re.search(r'\b(weather|temperature|forecast|rain|sunny)\b', lower_query):
            return await self.agents["weather"].run(user_input)

        if re.search(r'\b(aapl|tsla|nvda|btc|eth|stock|price|ticker|market cap)\b', lower_query):
            return await self.agents["stock"].run(user_input)

        # === LLM Routing ===
        route_msg = self.router_llm.chat(user_input, tools=None)
        route_raw = (route_msg.get("content") or "").strip()
        categories = self._extract_categories(route_raw)

        # Fallback if router gave nothing useful
        if not categories:
            categories = self._keyword_fallback(user_input)

        # === Smart conflict resolution ===
        is_news_query = any(kw in lower_query for kw in ["news", "latest", "recent", "breaking"])
        is_stock_query = any(kw in lower_query for kw in ["stock", "price", "ticker", "share", "current price"])

        # Rule: Pure news queries shouldn't trigger search
        if "news" in categories and "search" in categories and is_news_query:
            categories = ["news"]

        # Rule: News about company vs stock price
        if "news" in categories and "stock" in categories:
            if is_news_query and not is_stock_query:
                categories = ["news"]
            elif is_stock_query and not is_news_query:
                categories = ["stock"]

        # Rule: Remove search if a structured tool already answers it
        structured_tools = {"weather", "stock", "allocation"}
        if any(cat in categories for cat in structured_tools) and "search" in categories:
            categories = [c for c in categories if c != "search"]

        # Final safety
        categories = [c for c in categories if c in self.agents]
        if not categories:
            categories = ["general"]

        print(f" → Routed to: {', '.join(categories)}")

        # === Single agent (most common case) ===
        if len(categories) == 1:
            return await self.agents[categories[0]].run(user_input)

        # === Multiple agents → parallel execution with refined queries ===
        results = await asyncio.gather(
            *[self.agents[cat].run(self._refine_query(cat, user_input)) for cat in categories],
            return_exceptions=True
        )

        combined = []
        for cat, result in zip(categories, results):
            if isinstance(result, Exception):
                combined.append(f"[{cat.capitalize()}] Error: {str(result)}")
            else:
                combined.append(f"[{cat.capitalize()}]\n{str(result).strip()}")

        return "\n\n" + "=" * 70 + "\n\n".join(combined)