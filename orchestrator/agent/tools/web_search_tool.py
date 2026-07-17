"""
Web search tool using Tavily's search API.

Tavily returns already-cleaned page content (not raw HTML), which is why
it's the standard choice for agent web search over rolling your own
scraper — one API call gets you both the search and the extraction.
"""

from tavily import TavilyClient
from orchestrator.config import config


class WebSearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=config.TAVILY_API_KEY)

    def query(self, question: str, max_results: int = 3) -> dict:
        """
        Run a web search and return condensed context + source URLs.

        Returns: {"context": str, "sources": list[str]}
        """
        response = self.client.search(
            query=question,
            max_results=max_results,
            search_depth="basic",  # "advanced" costs more credits, not needed for most queries
        )

        results = response.get("results", [])
        context_parts = []
        sources = []

        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            context_parts.append(f"{title}: {content}")
            sources.append(url)

        return {"context": "\n\n".join(context_parts), "sources": sources}
