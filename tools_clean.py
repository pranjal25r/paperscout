"""
Tool 2: Clean, normalize, and dedupe collected paper records.
"""
import re
from langchain.tools import tool


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace for dedup comparison."""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


@tool
def clean_papers(papers: list[dict]) -> dict:
    """
    Clean and deduplicate a list of paper records.

    Args:
        papers: List of dicts from fetch_arxiv_papers, each with keys
                arxiv_id, title, authors, abstract, published, pdf_url

    Returns:
        Dict with keys:
            - cleaned: list of deduped, normalized paper dicts
            - stats: dict with counts of input, duplicates_removed, output
    """
    seen_ids = set()
    seen_titles = set()
    cleaned = []
    duplicates = 0

    for p in papers:
        arxiv_id = p.get("arxiv_id", "").strip()
        title = p.get("title", "").strip()
        norm_title = _normalize_title(title)

        # Skip malformed records
        if not arxiv_id or not title:
            continue

        # Dedup by arxiv_id OR normalized title
        if arxiv_id in seen_ids or norm_title in seen_titles:
            duplicates += 1
            continue

        seen_ids.add(arxiv_id)
        seen_titles.add(norm_title)

        abstract = p.get("abstract", "").strip()
        cleaned.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": p.get("authors", []),
            "abstract": abstract,
            "abstract_word_count": len(abstract.split()),
            "published": p.get("published", ""),
            "pdf_url": p.get("pdf_url", ""),
        })

    stats = {
        "input_count": len(papers),
        "duplicates_removed": duplicates,
        "output_count": len(cleaned),
    }

    return {"cleaned": cleaned, "stats": stats}


if __name__ == "__main__":
    # Quick standalone test with fake duplicate data
    sample = [
        {"arxiv_id": "1234.5678", "title": "Diffusion Models Are Great", "authors": ["A"], "abstract": "abc", "published": "2026-01-01", "pdf_url": "x"},
        {"arxiv_id": "1234.5678", "title": "Diffusion Models Are Great", "authors": ["A"], "abstract": "abc", "published": "2026-01-01", "pdf_url": "x"},
        {"arxiv_id": "9999.0001", "title": "  Diffusion   models are great  ", "authors": ["B"], "abstract": "def", "published": "2026-01-02", "pdf_url": "y"},
        {"arxiv_id": "8888.0002", "title": "Something Totally Different", "authors": ["C"], "abstract": "ghi", "published": "2026-01-03", "pdf_url": "z"},
    ]
    result = clean_papers.invoke({"papers": sample})
    print(result["stats"])
    for p in result["cleaned"]:
        print("-", p["title"])