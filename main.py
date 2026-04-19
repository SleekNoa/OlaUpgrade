# main.py
"""
OLA Agent - Main Entry Point
============================

This file serves as the central entry point for the entire project.

Two modes are supported:
1. **CLI Mode** (default for testing)
   → Runs the multi-agent system directly in the terminal with sample queries.
   → Perfect for debugging the orchestrator, agents, and tool calling loop.

2. **API Mode** (production / UI mode)
   → Starts the FastAPI server with the OpenAI-compatible endpoint.
   → Allows connecting Open WebUI, OlaChat, curl, LangChain, etc.

How to run:
- CLI testing:    `python main.py`
- API server:     `python main.py --api`   or set environment variable `RUN_MODE=api`

This design keeps everything in one file while clearly separating concerns.
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables (e.g. OPENAI_API_KEY if you use Phase 5 later)
load_dotenv()


async def run_cli_test():
    """
    CLI Test Runner - Full end-to-end test of the multi-agent system.

    What this does:
    1. Starts the MCP server (tools layer)
    2. Creates the MultiAgentOrchestrator
    3. Runs three test queries that cover:
       - Tool calling (weather)
       - Complex reasoning + tool use (task allocation)
       - Pure chat (no tools)

    This is the best way to verify everything works before starting the API server.
    """
    print("Starting OLA Agent CLI Test Mode")
    print("=" * 70)

    from tools.mcp_client import MCPClient
    from agent.multi_agent import MultiAgentOrchestrator

    async with MCPClient() as mcp:
        # Create the orchestrator with default model
        orchestrator = MultiAgentOrchestrator(mcp, model="llama3.1:8b")

        # Test queries covering different capabilities
        queries = [
            "What's the weather like in Marion, IA right now?",
            "Allocate tasks T1(db), T2(net), T3(any) to agents A1(db,cap=2), A2(net,cap=1)",
            "What's the capital of France?",
        ]

        for i, q in enumerate(queries, 1):
            print(f"\n{'=' * 70}")
            print(f"Test {i}/3 - USER: {q}")
            print("-" * 50)

            try:
                answer = await orchestrator.run(q)
                print(f"ANSWER: {answer}")
            except Exception as e:
                print(f"Error processing query: {e}")

        print("\n CLI Test completed successfully!")


def run_api_server():
    """
    Start the FastAPI server with OpenAI-compatible endpoints.

    This mode allows any OpenAI-compatible frontend (Open WebUI, etc.)
    to connect to your multi-agent system.

    The server will:
    - Start the MCP server once at boot (via lifespan)
    - Expose /v1/chat/completions
    - Provide /health and /v1/models for compatibility
    """
    print("   Starting OLA Agent API Server Mode")
    print("   OpenAI-compatible endpoint available at http://localhost:8000")
    print("   Connect Open WebUI or any client using base_url = http://localhost:8000")

    import uvicorn

    uvicorn.run(
        "api.server:app",  # Points to the FastAPI app in api/server.py
        host="0.0.0.0",  # Accessible from other machines (useful in Docker too)
        port=8000,
        reload=False,  # Set to True during active development
        log_level="info",
    )


def main():
    """
    Entry point that decides which mode to run.

    Priority:
    1. Command-line argument --api
    2. Environment variable RUN_MODE=api
    3. Default to CLI test mode (safest for development)
    """
    parser = argparse.ArgumentParser(description="OLA Agent - Multi-Agent System")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run in API server mode instead of CLI test mode"
    )
    args = parser.parse_args()

    # Check environment variable as fallback
    run_mode = os.getenv("RUN_MODE", "cli").lower()

    if args.api or run_mode == "api":
        run_api_server()
    else:
        # Run the async CLI test
        asyncio.run(run_cli_test())


if __name__ == "__main__":
    main()