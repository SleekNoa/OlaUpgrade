# main.py
"""Entry point for CLI testing or API server mode."""

import argparse
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()


async def run_cli_test():
    """Run a few representative prompts through the full MCP + agent stack."""
    from tools.mcp_client import MCPClient
    from agent.multi_agent import MultiAgentOrchestrator

    async with MCPClient() as mcp:
        orchestrator = MultiAgentOrchestrator(mcp)
        queries = [
            "What's the weather in Marion, IA?",
            "Allocate T1(db), T2(net), T3(any) to A1(skills=db,cap=2), A2(skills=net,cap=1)",
            "What is the capital of France?",
        ]
        for q in queries:
            print(f"\n{'=' * 60}\nUSER: {q}")
            answer = await orchestrator.run(q)
            print(f"ANSWER: {answer}")


def run_api_server():
    """Start the OpenAI-compatible FastAPI server."""
    import uvicorn
    print("Starting OLA Agent API on http://localhost:8000")
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    args = parser.parse_args()

    # --api wins, otherwise RUN_MODE decides between API and CLI.
    if args.api or os.getenv("RUN_MODE", "cli").lower() == "api":
        run_api_server()
    else:
        asyncio.run(run_cli_test())


if __name__ == "__main__":
    main()
