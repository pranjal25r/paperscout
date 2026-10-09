"""
Tool 3: Store cleaned paper records in a local SQLite database.
"""
import os
import sqlite3
import json
import uuid
from pathlib import Path
from langchain.tools import tool

# Override with the PAPERSCOUT_DB env var (e.g. a temp file for self-tests)
DB_PATH = Path(__file__).parent / "paperscout.db"


def _get_conn():
    conn = sqlite3.connect(os.getenv("PAPERSCOUT_DB") or DB_PATH)
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
    # Fetched batches wait here so paper data never has to pass through the LLM
    conn.execute("""
        CREATE TABLE IF NOT EXISTS staging (
            batch_id TEXT NOT NULL,
            arxiv_id TEXT,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            abstract_word_count INTEGER,
            published TEXT,
            pdf_url TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


def _purge_stale_staging():
    """Drop staged batches older than 1 hour (fetched but never stored)."""
    conn = _get_conn()
    conn.execute("DELETE FROM staging WHERE created_at < datetime('now', '-1 hour')")
    conn.commit()
    conn.close()


def _stage_papers(papers: list[dict]) -> str:
    """Insert raw paper records into staging under a new batch_id and return it."""
    batch_id = str(uuid.uuid4())
    conn = _get_conn()
    conn.executemany(
        """INSERT INTO staging
           (batch_id, arxiv_id, title, authors, abstract, abstract_word_count, published, pdf_url)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                batch_id,
                p.get("arxiv_id", ""),
                p.get("title", ""),
                json.dumps(p.get("authors", [])),
                p.get("abstract", ""),
                p.get("abstract_word_count", 0),
                p.get("published", ""),
                p.get("pdf_url", ""),
            )
            for p in papers
        ],
    )
    conn.commit()
    conn.close()
    return batch_id


@tool
def store_papers(batch_id: str) -> dict:
    """
    Store a staged batch of paper records into a local SQLite database,
    skipping records that already exist (by arxiv_id). The batch is removed
    from staging afterwards.

    Args:
        batch_id: Batch identifier returned by fetch_arxiv_papers (run
                  clean_papers on it first)

    Returns:
        Dict with keys: inserted, skipped_existing, total_in_db
    """
    conn = _get_conn()
    cur = conn.cursor()

    rows = cur.execute(
        "SELECT arxiv_id, title, authors, abstract, abstract_word_count, published, pdf_url "
        "FROM staging WHERE batch_id = ? ORDER BY rowid",
        (batch_id,),
    ).fetchall()

    inserted = 0
    skipped = 0

    for r in rows:
        cur.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (r[0],))
        if cur.fetchone():
            skipped += 1
            continue

        cur.execute(
            """INSERT INTO papers
               (arxiv_id, title, authors, abstract, abstract_word_count, published, pdf_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            r,
        )
        inserted += 1

    cur.execute("DELETE FROM staging WHERE batch_id = ?", (batch_id,))
    conn.commit()
    total = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()

    result = {"inserted": inserted, "skipped_existing": skipped, "total_in_db": total}
    if not rows:
        result["note"] = "No staged papers found for this batch_id."
    return result


@tool
def query_stored_papers(limit: int = 10) -> list[dict]:
    """
    Retrieve stored papers from the local database, most recent first.

    Args:
        limit: Max number of records to return

    Returns:
        List of dicts with keys: arxiv_id, title, authors, published
    """
    conn = _get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT arxiv_id, title, authors, published "
        "FROM papers ORDER BY published DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    return [
        {
            "arxiv_id": r[0],
            "title": r[1],
            "authors": json.loads(r[2]) if r[2] else [],
            "published": r[3],
        }
        for r in rows
    ]


if __name__ == "__main__":
    import tempfile

    # Quick standalone test against a throwaway DB
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["PAPERSCOUT_DB"] = str(Path(tmp_dir.name) / "test.db")

    sample = [
        {"arxiv_id": "1111.1111", "title": "Test Paper One", "authors": ["X"], "abstract": "abc", "abstract_word_count": 1, "published": "2026-01-01", "pdf_url": "url1"},
        {"arxiv_id": "2222.2222", "title": "Test Paper Two", "authors": ["Y"], "abstract": "def", "abstract_word_count": 1, "published": "2026-01-02", "pdf_url": "url2"},
    ]
    result = store_papers.invoke({"batch_id": _stage_papers(sample)})
    print("Store result:", result)

    # Stage and store the same records again to prove skip-existing works
    result2 = store_papers.invoke({"batch_id": _stage_papers(sample)})
    print("Second store (should skip both):", result2)

    stored = query_stored_papers.invoke({"limit": 5})
    print("Stored papers:")
    for p in stored:
        print("-", p["title"], p["arxiv_id"])

    tmp_dir.cleanup()
