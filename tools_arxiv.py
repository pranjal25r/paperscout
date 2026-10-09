"""
Tool 1: Fetch papers from arXiv by search query.
"""
import re

import arxiv
from langchain.tools import tool

from tools_storage import _stage_papers, _purge_stale_staging


def _build_query(query: str) -> str:
    """Require every term; queries already using arXiv syntax pass through unchanged."""
    if ":" in query or re.search(r"\b(AND|OR)\b", query):
        return query
    terms = [t.strip("\"'") for t in query.split()]
    return " AND ".join(f"all:{t}" for t in terms if t)


@tool
def fetch_arxiv_papers(query: str, max_results: int = 5) -> dict:
    """
    Search arXiv for papers matching a query and stage them for cleaning and
    storage. Use the number of papers the user asked for as max_results. Call
    once per request.

    Args:
        query: Search terms, e.g. "diffusion transformer" or "retrieval augmented generation"
        max_results: Number of papers to fetch (default 5, clamped to 1-20)

    Returns:
        Dict with keys: batch_id (pass to clean_papers and store_papers),
        fetched (count), titles (truncated to 80 chars)
    """
    _purge_stale_staging()
    max_results = max(1, min(20, max_results))
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=_build_query(query),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title.strip().replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip().replace("\n", " "),
            "published": result.published.strftime("%Y-%m-%d"),
            "pdf_url": result.pdf_url,
        })

    batch_id = _stage_papers(papers)
    return {
        "batch_id": batch_id,
        "fetched": len(papers),
        "titles": [p["title"][:80] for p in papers],
    }


if __name__ == "__main__":
    import os
    import tempfile
    from pathlib import Path

    # Quick standalone test against a throwaway DB
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["PAPERSCOUT_DB"] = str(Path(tmp_dir.name) / "test.db")

    for query, n in [("large language model safety", 5), ("diffusion transformer", 3)]:
        print(f"query sent: {_build_query(query)}")
        result = fetch_arxiv_papers.invoke({"query": query, "max_results": n})
        print(f"batch_id: {result['batch_id']}, fetched: {result['fetched']}")
        for t in result["titles"]:
            print(f"- {t}")

    tmp_dir.cleanup()
