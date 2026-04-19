# OLA Agent (Orchestrated LLM Agent)

**OLA Agent** is a modular, production-ready multi-agent system that intelligently routes user queries to specialized agents backed by a robust tool-calling architecture.

It combines:
- Smart routing (LLM + keyword fallbacks)
- Specialist agents for different domains
- Native tool calling via **MCP** (Model Context Protocol)
- OpenAI-compatible API server for seamless integration

## Features

- **Intelligent Orchestrator**: LLM-based router with strong keyword overrides and conflict resolution
- **Specialist Agents**: Dedicated agents for Weather, News, Stocks, Web Search, Task Allocation, and General knowledge
- **Robust Tool Loop**: BaseAgent with loop detection, repeated call protection, and fake tool-call handling
- **MCP Architecture**: Clean separation between agents and tools using FastMCP
- **OpenAI-Compatible API**: `/v1/chat/completions` endpoint – works with Cursor, Continue.dev, etc.
- **Multi-LLM Support**: Ollama (default), OpenAI, Groq, LM Studio, and any OpenAI-compatible endpoint
- **Production Ready**: Shared MCP lifecycle, clean logging to stderr, graceful error handling

## Project Structure
OlaUpgrade/
├── main.py                    # CLI test mode + API server entry point
├── agent/
│   ├── base_agent.py          # Core tool-calling loop
│   ├── multi_agent.py         # Router + specialist orchestrator
│   └── init.py
├── api/
│   └── server.py              # FastAPI OpenAI-compatible server
├── llm/
│   ├── base.py
│   ├── factory.py
│   ├── ollama_client.py
│   └── openai_client.py
├── tools/
│   └── mcp_client.py          # MCP client bridge
├── mcp-server/
│   ├── mcp_server.py          # FastMCP tool server
│   └── utils/
│       └── logging_config.py
├── providers/
│   ├── news.py                # NewsAPI integration
│   ├── openmeteo.py           # Weather with smart US state geocoding
│   ├── search.py              # DuckDuckGo search
│   └── stocks.py              # yfinance with robust session handling
├── .env.example
├── requirements.txt
└── README.md
text## Quick Start

### 1. Installation

```bash
cd OlaUpgrade

# Recommended: create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
2. Configuration
Copy .env.example to .env and configure:
env# LLM Settings
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b

# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini

# API Keys (only for the providers you use)
NEWS_API_KEY=your_newsapi_key_here
# OPENAI_API_KEY=sk-...

# Optional
RUN_MODE=cli        # or "api"
3. Running the Agent
CLI Test Mode (great for development):
Bashpython main.py
API Server Mode:
Bashpython main.py --api
# or set RUN_MODE=api in .env
The server will be available at: http://localhost:8000
Useful endpoints:

GET / → Simple status page
GET /health
GET /v1/models
POST /v1/chat/completions → OpenAI compatible

Supported Tools

get_weather(city) – Current weather (excellent support for "Marion, IA")
get_news(topic) – Recent news via NewsAPI
search_web(query) – Web search via DuckDuckGo
get_stock(ticker) – Stock & market data via yfinance
allocate_tasks(agents, tasks, constraints?) – Resource allocation engine

Example Queries

"What's the weather in Marion, IA?"
"Allocate T1(db), T2(net), T3(any) to A1(skills=db,cap=2), A2(skills=net,cap=1)"
"Latest news about artificial intelligence"
"Current price of TSLA"
"What is the capital of France?"

Development
Adding a New Tool

Implement logic in providers/
Register it with @mcp.tool() in mcp-server/mcp_server.py
(Optional) Add a specialist agent in agent/multi_agent.py

Logging
MCP logs go to stderr to keep the JSON-RPC stream clean.
Tech Stack

API: FastAPI + Uvicorn
Tools: FastMCP + MCP Python SDK
LLMs: Ollama, OpenAI SDK (compatible with Groq, LM Studio, vLLM, etc.)
Providers: Open-Meteo, NewsAPI, DuckDuckGo, yfinance