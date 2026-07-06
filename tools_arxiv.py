"""
Tool 1: Fetch papers from arXiv by search query.
"""
import arxiv
from langchain.tools import tool


@tool
def fetch_arxiv_papers(query: str, max_results: int = 10) -> list[dict]:
    """
    Search arXiv for papers matching a query and return structured metadata.

    Args:
        query: Search terms, e.g. "diffusion transformer" or "retrieval augmented generation"
        max_results: Number of papers to fetch (default 10)

    Returns:
        List of dicts with keys: title, authors, abstract, published, pdf_url, arxiv_id
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
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

    return papers


if __name__ == "__main__":
    # Quick standalone test
    results = fetch_arxiv_papers.invoke({"query": "diffusion transformer", "max_results": 3})
    for r in results:
        print(f"- {r['title']} ({r['published']})")