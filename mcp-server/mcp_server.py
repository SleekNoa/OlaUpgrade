# mcp-server/mcp_server.py
"""
MCP tool server for the OLA agent.

Important stdio rule:
- MCP responses travel over stdout as JSON-RPC.
- Any accidental print output from this process can corrupt that stream.
- Logging is configured to write to stderr instead.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

# Make project root importable so sibling packages like providers/ resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from providers.openmeteo import get_weather as openmeteo_get_weather
from utils.logging_config import configure_logging

configure_logging()
log = logging.getLogger("mcp-server")
mcp = FastMCP(name="ola-tools")


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


class Constraints(BaseModel):
    """Optional global allocation constraints."""
    max_tasks_per_agent: Optional[int] = Field(default=None, ge=1)


@mcp.tool()
def get_weather(city: str) -> dict:
    """Get current weather for a city. Returns temp_c, conditions, provider."""
    log.info("get_weather(%r)", city)
    res = openmeteo_get_weather(city.strip())
    if not res:
        raise RuntimeError(f"Weather fetch failed for '{city}'")
    return {"temp_c": res.temp_c, "conditions": res.conditions, "provider": res.provider}


@mcp.tool()
def allocate_tasks(
    agents: list[dict],
    tasks: list[dict],
    constraints: dict | None = None,
) -> dict:
    """
    Allocate tasks to agents using a simple least-loaded strategy.

    Rules:
    - Respect required_skill when present
    - Respect per-agent capacity
    - Respect max_tasks_per_agent when provided
    """
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
    for t in parsed_tasks:
        candidates = sorted(
            [a.id for a in parsed_agents if eligible(a.id, t.required_skill)],
            key=lambda aid: load[aid],
        )
        if candidates:
            chosen = candidates[0]
            assignments.append({"task_id": t.id, "agent_id": chosen})
            load[chosen] += 1
        else:
            unassigned.append(t.id)

    return {"assignments": assignments, "unassigned": unassigned}


if __name__ == "__main__":
    # FastMCP serves over stdio here so the client can attach as a subprocess.
    mcp.run()
