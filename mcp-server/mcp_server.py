# mcp-server/mcp_server.py
r"""
MCP Server (Model Context Protocol)
====================================
This is the tool execution layer. It exposes functions that LLMs can call.

We use FastMCP because it provides a clean, standard way to define tools
that can be discovered and called via stdio JSON-RPC.

Key benefits:
- Tools are defined with normal Python type hints
- FastMCP automatically generates proper JSON schemas
- Runs as a separate process (good isolation)
"""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import logging
from providers.openmeteo import get_weather as openmeteo_get_weather


log = logging.getLogger("mcp-server")

# Initialize the MCP server
mcp = FastMCP(name="ola-minimal-tools")


# ===== Input Schemas (for type safety and auto-generated tool descriptions) =====
class WeatherArgs(BaseModel):
    """Input schema for get_weather tool."""
    city: str = Field(...,
                      description="City name with optional state/country",
                      examples=["Marion, IA", "Austin, TX", "Paris"]
    )


class Agent(BaseModel):
    """Represents one agent/resource that can be assigned tasks."""
    id: str = Field(...,
                    description="Unique identifier for the agent",
                    examples=["A1", "db-specialist", "Alice"]
    )
    skills: List[str] = Field(
        default_factory=list,
        description="List of skills this agent has",
        examples=[["db"], ["network", "security"]]
    )
    capacity: int = Field(
        default=1,
        ge=1,
        description="Maximum number of tasks this agent can handle simultaneously",
        examples=[1, 2, 5]
    )

class Task(BaseModel):
    """Represents a single task that needs to be assigned."""
    id: str = Field(
        ...,
        description="Unique identifier for the task",
        examples=["T1", "backend-setup", "report-generation"]
    )
    required_skill: Optional[str] = Field(
        default=None,
        description="Specific skill required for this task. If None, any agent can take it.",
        examples=["db", "network", None]
    )
    effort: int = Field(
        default=1,
        ge=1,
        description="Effort level or estimated workload units for this task",
        examples=[1, 2, 3]
    )

class Constraints(BaseModel):
    """Global constraints for the task allocation."""
    max_tasks_per_agent: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional hard limit on tasks per agent (overrides individual capacity if stricter)",
        examples=[None, 3, 5]
    )

class AllocateArgs(BaseModel):
    """Input schema for allocate_tasks tool."""
    agents: List[Agent] = Field(
        ...,
        description="List of available agents",
        min_length=1
    )
    tasks: List[Task] = Field(
        ...,
        description="List of tasks to be allocated",
        min_length=1
    )
    constraints: Optional[Constraints] = Field(
        default=None,
        description="Optional global constraints for allocation"
    )

# ===== Tools - CORRECT FastMCP syntax =====
@mcp.tool()                    # Note: Use @mcp.tool() with type hints only. Do NOT use schema= argument.
def get_weather(city: str) -> dict:
    """
    Get current weather for a city using Open-Meteo API.

    This tool is exposed to the LLM via MCP. The LLM will call it when
    it needs weather information.
    """
    log.info(f"get_weather called with city={city!r}")

    res = openmeteo_get_weather(city.strip())
    if not res:
        raise RuntimeError(f"Weather fetch failed for city: '{city}'")

    return {
        "temp_c": res.temp_c,
        "conditions": res.conditions,
        "provider": res.provider
    }


@mcp.tool()
def allocate_tasks(
    agents: list[dict],
    tasks: list[dict],
    constraints: dict | None = None,
) -> dict:
    """
    Allocate tasks to agents based on skills, capacity, and constraints.

    This is a classic resource allocation problem. The LLM can call this
    when the user asks to assign work to team members.
    """
    log.info("allocate_tasks called")

    # Parse raw dicts into proper Pydantic models for safety

    parsed_agents = [Agent(**a) for a in agents]
    parsed_tasks = [Task(**t) for t in tasks]
    con = Constraints(**(constraints or {}))

    # Tracking structures

    load = {a.id: 0 for a in parsed_agents}
    skills_map = {a.id: set(a.skills) for a in parsed_agents}
    cap_map = {a.id: a.capacity for a in parsed_agents}

    def eligible(aid: str, skill: str | None) -> bool:
        """Check if an agent can take a task."""
        if skill and skill not in skills_map[aid]:
            return False
        limit = con.max_tasks_per_agent
        return (load[aid] < cap_map[aid] and
                (limit is None or load[aid] < limit))

    assignments = []
    unassigned = []

    for t in parsed_tasks:
        candidates = sorted(
            [a.id for a in parsed_agents if eligible(a.id, t.required_skill)],
            key = lambda aid: load[aid],
        )
        if candidates:
            chosen = candidates[0]
            assignments.append({"task_id": t.id, "agent_id": chosen})
            load[chosen] += 1
        else:
            unassigned.append(t.id)

    return  {"assignments": assignments,
             "unassigned": unassigned
    }

if __name__ == "__main__":
    # Run the MCP server over stdio (this is how the client connects)
    mcp.run()   # pure stdio JSON-RPC



