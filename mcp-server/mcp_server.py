# mcp-server/mcp_server.py
"""
MCP tool server for the OLA agent.

Important stdio rule:
- MCP responses travel over stdout as JSON-RPC.
- Any accidental print output from this process can corrupt that stream.
- Logging is configured to write to stderr instead.

This server exposes the shared tool layer used by the API/orchestrator and by external MCP clients.
"""

from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Make project root importable so sibling packages like providers/ resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from providers.news import get_news as news_get_news
from providers.openmeteo import get_weather as openmeteo_get_weather
from providers.search import search_web as ddg_search
from providers.stocks import get_stock as stocks_get_stock
from utils.logging_config import configure_logging

configure_logging()
log = logging.getLogger("mcp-server")

mcp = FastMCP(name="ola-tools")


# ==================== GLOBAL ERROR HELPER ====================
def _make_config_error_result(tool_name: str, message: str) -> dict:
    """Consistent graceful fallback used by all tools."""
    return {
        "success": False,
        "error_type": "CONFIG_MISSING",
        "error": message,
        "is_fallback": True,
        "fallback_reason": message,
        "tool_name": tool_name,
        "data": None,
        "timestamp": datetime.now().isoformat()
    }


class Constraints(BaseModel):
    """Optional global allocation constraints."""
    max_tasks_per_agent: Optional[int] = Field(default=None, ge=1)


class Agent(BaseModel):
    """One worker/resource available for assignment."""
    id: str
    skills: List[str] = Field(default_factory=list)
    capacity: int = Field(default=1, ge=1)


class Task(BaseModel):
    """One unit of work to allocate."""
    id: str
    required_skill: Optional[str] = None
    effort: int = Field(default=1, ge=1)


@mcp.tool()
def get_weather(city: str) -> dict:
    """Get current weather for a specific city."""
    log.info("get_weather(%r)", city)
    try:
        res = openmeteo_get_weather(city.strip())
        if not res:
            raise RuntimeError(f"Weather fetch failed for '{city}'")
        return {
            "temp_c": res.temp_c,
            "conditions": res.conditions,
            "provider": res.provider
        }
    except Exception as e:
        log.warning("Weather failed: %s", e)
        return {
            "temp_c": None,
            "conditions": "Service unavailable",
            "provider": "fallback",
            "error": None
        }


@mcp.tool()
def allocate_tasks(
    agents: list[dict],
    tasks: list[dict],
    constraints: dict | None = None,
) -> dict:
    """
    Allocate tasks to agents using a simple least-loaded strategy.
    Each agent needs: id, skills[], capacity.
    Each task needs: id, required_skill (optional).
    """
    import json as _json

    # LLM sometimes passes JSON strings instead of parsed lists
    if isinstance(agents, str):
        agents = _json.loads(agents)
    if isinstance(tasks, str):
        tasks = _json.loads(tasks)

    # Normalise task dicts: LLM sometimes uses 'name' instead of 'id'
    normalised_tasks = []
    for t in tasks:
        if isinstance(t, dict) and "id" not in t and "name" in t:
            t = {**t, "id": t["name"]}
        normalised_tasks.append(t)
    tasks = normalised_tasks

    # Normalise agent dicts: same issue
    normalised_agents = []
    for a in agents:
        if isinstance(a, dict) and "id" not in a and "name" in a:
            a = {**a, "id": a["name"]}
        normalised_agents.append(a)
    agents = normalised_agents

    log.info("allocate_tasks()")
    parsed_agents = [Agent(**a) for a in agents]
    parsed_tasks = [Task(**t) for t in tasks]
    con = Constraints(**(constraints or {}))

    load = {a.id: 0 for a in parsed_agents}
    skills_map = {a.id: set(a.skills) for a in parsed_agents}
    cap_map = {a.id: a.capacity for a in parsed_agents}

    def eligible(aid: str, skill: Optional[str]) -> bool:
        if skill and skill not in skills_map[aid]:
            return False
        limit = con.max_tasks_per_agent
        return load[aid] < cap_map[aid] and (limit is None or load[aid] < limit)

    assignments, unassigned = [], []
    for task in parsed_tasks:
        candidates = sorted(
            [agent.id for agent in parsed_agents if eligible(agent.id, task.required_skill)],
            key=lambda aid: load[aid],
        )
        if candidates:
            chosen = candidates[0]
            assignments.append({"task_id": task.id, "agent_id": chosen})
            load[chosen] += 1
        else:
            unassigned.append(task.id)

    return {"assignments": assignments, "unassigned": unassigned}


@mcp.tool()
def get_news(topic: str, max_results: int = 5) -> dict:
    """Get current news articles for a topic."""
    log.info("get_news(%r)", topic)
    max_results = min(max_results, 10)

    try:
        res = news_get_news(topic.strip(), max_results=max_results)

        if res.error:
            log.error("News provider returned error: %s", res.error)
            # More informative fallback
            return {
                "topic": topic,
                "count": 0,
                "articles": [],
                "error": res.error,
                "success": False,
                "message": "News service temporarily unavailable. Please try again shortly."
            }

        return {
            "topic": res.topic,
            "count": len(res.articles),
            "articles": res.articles,
            "success": True,
            "error": None  # ← IMPORTANT
        }

    except Exception as e:
        log.exception("Unexpected error in get_news tool")
        return _make_config_error_result(
            "get_news",
            "News service encountered an unexpected error."
        )


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo."""
    log.info("search_web(%r)", query)
    max_results = min(max_results, 10)
    try:
        res = ddg_search(query.strip(), max_results=max_results)
        if res.error:
            raise RuntimeError(f"Search failed: {res.error}")
        return {
            "query": res.query,
            "count": len(res.results),
            "results": res.results
        }
    except Exception as e:
        log.warning("Search failed: %s", e)
        return _make_config_error_result(
            "search_web",
            "Web search provider encountered an issue (possible missing configuration)."
        )


@mcp.tool()
def get_stock(ticker: str) -> dict:
    """Get current stock/market data for a ticker symbol."""
    log.info("get_stock(%r)", ticker)
    try:
        res = stocks_get_stock(ticker.strip())
        if res.error:
            raise RuntimeError(f"Stock fetch failed for '{ticker}': {res.error}")
        return {
            "ticker": res.ticker,
            "company": res.company,
            "price": res.price,
            "currency": res.currency,
            "change_pct": res.change_pct,
            "market_cap": res.market_cap,
        }
    except Exception as e:
        log.warning("Stock tool failed for '%s': %s", ticker, e)
        return _make_config_error_result(
            "get_stock",
            "Stock data provider not configured or temporarily unavailable."
        )


if __name__ == "__main__":
    # FastMCP serves over stdio here so the client can attach as a subprocess.
    mcp.run()