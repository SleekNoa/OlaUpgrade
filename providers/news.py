# providers/news.py
"""
News provider for the OLA tool layer.

Uses NewsAPI to fetch current news articles for a topic. This provider stays
separate from the MCP tool so we keep API integration logic isolated and easy
to test or swap later.
"""

# providers/news.py

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

log = logging.getLogger("news-provider")


@dataclass
class NewsResult:
    topic: str
    articles: List[dict] = field(default_factory=list)
    error: str | None = None


def _build_query(topic: str) -> tuple[str, str, str | None]:
    """
    Build a smarter NewsAPI query.
    Returns: (query, sort_by, from_param)
    """
    q = topic.strip()
    lower = q.lower()

    sort_by = "relevancy"
    from_param = None

    # 🔥 Recency detection
    if any(word in lower for word in [
        "latest", "recent", "news", "today", "now",
        "who won", "winner", "super bowl", "election"
    ]):
        sort_by = "publishedAt"
        from_param = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")

    # 🔥 Topic expansions (clean + scalable)
    expansions = {
        "artificial intelligence": '"artificial intelligence" OR AI OR "machine learning"',
        "ai": '"artificial intelligence" OR AI OR "machine learning"',
        "tesla": 'Tesla OR "Elon Musk" OR EV OR "electric vehicles"',
        "super bowl": "Super Bowl winner 2026",
    }

    for key, value in expansions.items():
        if key in lower:
            q = value
            sort_by = "publishedAt"
            break

    return q, sort_by, from_param


def get_news(topic: str, max_results: int = 5) -> NewsResult:
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        log.warning("NEWS_API_KEY not set in environment")
        return NewsResult(
            topic=topic,
            error="NEWS_API_KEY not set in environment"
        )

    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=api_key)

        # ✅ Use improved query builder
        q, sort_by, from_param = _build_query(topic)

        log.info("Fetching news for query: '%s' (sort: %s)", q, sort_by)

        response = client.get_everything(
            q=q,
            language="en",
            sort_by=sort_by,
            from_param=from_param,
            page_size=min(max_results, 20),
        )

        articles = []
        for article in response.get("articles", [])[:max_results]:
            articles.append({
                "title": article.get("title", "No title"),
                "source": article.get("source", {}).get("name", "Unknown"),
                "published": article.get("publishedAt", "")[:10],
                "description": (article.get("description") or article.get("content", ""))[:280],
                "url": article.get("url", ""),
            })

        # ⚠️ Fallback if API returns nothing useful
        if not articles:
            log.warning("No articles returned for topic: %s", topic)
            return NewsResult(
                topic=topic,
                error="No articles found for this topic"
            )

        log.info("Successfully fetched %d articles for topic: %s", len(articles), topic)
        return NewsResult(topic=topic, articles=articles)

    except ImportError:
        log.error("newsapi-python package is not installed")
        return NewsResult(
            topic=topic,
            error="newsapi-python package not installed"
        )

    except Exception as exc:
        log.exception("Failed to fetch news for topic '%s'", topic)
        return NewsResult(
            topic=topic,
            error=str(exc)
        )