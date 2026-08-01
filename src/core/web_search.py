import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def perform_web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Executes a DuckDuckGo text search for the given query."""
    if not query or not query.strip():
        return []

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = list(DDGS().text(query.strip(), max_results=max_results))
        return results
    except Exception as err:
        logger.error(f"DuckDuckGo web search error for query '{query}': {err}")
        return []


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Formats DuckDuckGo search result objects into markdown text."""
    if not results:
        return "No web search results found."

    formatted = ["### Live Web Search Results\n"]
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        url = item.get("href") or item.get("link") or "#"
        body = item.get("body") or item.get("snippet") or ""
        formatted.append(f"{idx}. [{title}]({url})\n   {body}\n")

    return "\n".join(formatted)
