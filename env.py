# .env
LLM_PROVIDER=ollama          # ollama | openai | groq | lmstudio
LLM_MODEL=llama3.1:8b        # override model name

RUN_MODE=api

# Only needed if LLM_PROVIDER=openai or compatible:
# OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=http://localhost:11434/v1   # LM Studio / vLLM / Groq