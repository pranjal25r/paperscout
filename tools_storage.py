"""
Tool 3: Store cleaned paper records in a local SQLite database.
"""
import sqlite3
import json
from pathlib import Path
from langchain.tools import tool

DB_PATH = Path(__file__).parent / "paperscout.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            abstract_word_count INTEGER,
            published TEXT,
            pdf_url TEXT
        )
    """)
    return conn


@tool
def store_papers(papers: list[dict]) -> dict:
    """
    Store cleaned paper records into a local SQLite database, skipping
    records that already exist (by arxiv_id).

    Args:
        papers: List of cleaned paper dicts (from clean_papers), each with
                keys arxiv_id, title, authors, abstract, abstract_word_count,
                published, pdf_url

    Returns:
        Dict with keys: inserted, skipped_existing, total_in_db
    """
    conn = _get_conn()
    cur = conn.cursor()

    inserted = 0
    skipped = 0

    for p in papers:
        cur.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (p["arxiv_id"],))
        if cur.fetchone():
            skipped += 1
            continue

        cur.execute(
            """INSERT INTO papers
               (arxiv_id, title, authors, abstract, abstract_word_count, published, pdf_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                p["arxiv_id"],
                p["title"],
                json.dumps(p.get("authors", [])),
                p.get("abstract", ""),
                p.get("abstract_word_count", 0),
                p.get("published", ""),
                p.get("pdf_url", ""),
            ),
        )
        inserted += 1

    conn.commit()
    total = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()

    return {"inserted": inserted, "skipped_existing": skipped, "total_in_db": total}


@tool
def query_stored_papers(limit: int = 10) -> list[dict]:
    """
    Retrieve stored papers from the local database, most recent first.

    Args:
        limit: Max number of records to return

    Returns:
        List of paper dicts
    """
    conn = _get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT arxiv_id, title, authors, abstract, published, pdf_url "
        "FROM papers ORDER BY published DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    return [
        {
            "arxiv_id": r[0],
            "title": r[1],
            "authors": json.loads(r[2]) if r[2] else [],
            "abstract": r[3],
            "published": r[4],
            "pdf_url": r[5],
        }
        for r in rows
    ]


if __name__ == "__main__":
    # Quick standalone test
    sample = [
        {"arxiv_id": "1111.1111", "title": "Test Paper One", "authors": ["X"], "abstract": "abc", "abstract_word_count": 1, "published": "2026-01-01", "pdf_url": "url1"},
        {"arxiv_id": "2222.2222", "title": "Test Paper Two", "authors": ["Y"], "abstract": "def", "abstract_word_count": 1, "published": "2026-01-02", "pdf_url": "url2"},
    ]
    result = store_papers.invoke({"papers": sample})
    print("Store result:", result)

    # Insert same records again to prove skip-existing works
    result2 = store_papers.invoke({"papers": sample})
    print("Second store (should skip both):", result2)

    stored = query_stored_papers.invoke({"limit": 5})
    print("Stored papers:")
    for p in stored:
        print("-", p["title"], p["arxiv_id"])