"""
Tool 2: Clean, normalize, and dedupe collected paper records.
"""
import re
from langchain.tools import tool

from tools_storage import _get_conn


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace for dedup comparison."""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


@tool
def clean_papers(batch_id: str) -> dict:
    """
    Clean and deduplicate a staged batch of paper records in place.

    Args:
        batch_id: Batch identifier returned by fetch_arxiv_papers

    Returns:
        Dict with keys: batch_id, input_count, duplicates_removed, output_count
    """
    conn = _get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT rowid, arxiv_id, title, abstract FROM staging WHERE batch_id = ? ORDER BY rowid",
        (batch_id,),
    ).fetchall()

    seen_ids = set()
    seen_titles = set()
    kept = 0
    duplicates = 0

    for rowid, arxiv_id, title, abstract in rows:
        arxiv_id = (arxiv_id or "").strip()
        title = (title or "").strip()
        norm_title = _normalize_title(title)

        # Skip malformed records
        if not arxiv_id or not title:
            cur.execute("DELETE FROM staging WHERE rowid = ?", (rowid,))
            continue

        # Dedup by arxiv_id OR normalized title
        if arxiv_id in seen_ids or norm_title in seen_titles:
            duplicates += 1
            cur.execute("DELETE FROM staging WHERE rowid = ?", (rowid,))
            continue

        seen_ids.add(arxiv_id)
        seen_titles.add(norm_title)

        abstract = (abstract or "").strip()
        cur.execute(
            "UPDATE staging SET arxiv_id = ?, title = ?, abstract = ?, abstract_word_count = ? "
            "WHERE rowid = ?",
            (arxiv_id, title, abstract, len(abstract.split()), rowid),
        )
        kept += 1

    conn.commit()
    conn.close()

    result = {
        "batch_id": batch_id,
        "input_count": len(rows),
        "duplicates_removed": duplicates,
        "output_count": kept,
    }
    if not rows:
        result["note"] = "No staged papers found for this batch_id."
    return result


if __name__ == "__main__":
    import os
    import tempfile
    from pathlib import Path

    from tools_storage import _stage_papers

    # Quick standalone test with fake duplicate data, against a throwaway DB
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["PAPERSCOUT_DB"] = str(Path(tmp_dir.name) / "test.db")

    sample = [
        {"arxiv_id": "1234.5678", "title": "Diffusion Models Are Great", "authors": ["A"], "abstract": "abc", "published": "2026-01-01", "pdf_url": "x"},
        {"arxiv_id": "1234.5678", "title": "Diffusion Models Are Great", "authors": ["A"], "abstract": "abc", "published": "2026-01-01", "pdf_url": "x"},
        {"arxiv_id": "9999.0001", "title": "  Diffusion   models are great  ", "authors": ["B"], "abstract": "def", "published": "2026-01-02", "pdf_url": "y"},
        {"arxiv_id": "8888.0002", "title": "Something Totally Different", "authors": ["C"], "abstract": "ghi", "published": "2026-01-03", "pdf_url": "z"},
    ]
    batch_id = _stage_papers(sample)
    result = clean_papers.invoke({"batch_id": batch_id})
    print(result)

    conn = _get_conn()
    for (title,) in conn.execute("SELECT title FROM staging WHERE batch_id = ? ORDER BY rowid", (batch_id,)):
        print("-", title)
    conn.close()

    tmp_dir.cleanup()
