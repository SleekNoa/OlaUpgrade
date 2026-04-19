# providers/search.py
"""DuckDuckGo-powered web search provider for current-information lookups."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchResult:
    """Structured search response returned to the MCP layer."""
    query: str
    results: List[dict] = field(default_factory=list)
    error: str | None = None


def search_web(query: str, max_results: int = 5) -> SearchResult:
    """
    Search the web via DuckDuckGo.
    Uses the newer 'ddgs' package (recommended) with fallback.
    """
    max_results = min(max_results, 10)

    try:
        # Try the new recommended package first
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = []
                for row in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": row.get("title", ""),
                        "url": row.get("href", ""),
                        "snippet": row.get("body", ""),
                    })
                if results:
                    return SearchResult(query=query, results=results)
        except ImportError:
            pass

        # Fallback to old package if ddgs is not installed
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": row.get("title", ""),
                    "url": row.get("href", ""),
                    "snippet": row.get("body", ""),
                })

        return SearchResult(query=query, results=results)

    except Exception as exc:
        error_str = str(exc)
        if "429" in error_str or "rate limit" in error_str.lower():
            error_str = "Rate limited - try again in a few seconds"
        return SearchResult(query=query, error=error_str)