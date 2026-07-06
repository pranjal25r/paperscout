"""
Tool 4: Build a FAISS index over stored papers and answer questions
using retrieval-augmented generation over the agent's own collection.
"""
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.tools import tool

from tools_storage import _get_conn

INDEX_DIR = Path(__file__).parent / "index"
INDEX_DIR.mkdir(exist_ok=True)
FAISS_PATH = INDEX_DIR / "papers.faiss"
META_PATH = INDEX_DIR / "papers_meta.pkl"

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _fetch_all_papers() -> list[dict]:
    conn = _get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT arxiv_id, title, authors, abstract, published, pdf_url FROM papers"
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


@tool
def build_index(reason: str = "refresh index") -> dict:
    """
    Build (or rebuild) a FAISS index over all papers currently stored in
    the database. Must be called at least once before query_collection
    can be used, and again any time new papers are added.

    Args:
        reason: Brief note on why the index is being built (not used in logic,
                only present to satisfy the tool-calling schema)

    Returns:
        Dict with keys: indexed_count, index_path
    """
    papers = _fetch_all_papers()
    if not papers:
        return {"indexed_count": 0, "index_path": str(FAISS_PATH), "note": "No papers in DB yet."}

    embedder = _get_embedder()
    # Embed title + abstract together for richer semantic signal
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(papers, f)

    return {"indexed_count": len(papers), "index_path": str(FAISS_PATH)}


@tool
def query_collection(question: str, top_k: int = 3, min_score: float = 0.25) -> dict:
    """
    Retrieve the most relevant stored papers for a natural-language question,
    using semantic similarity search over the FAISS index built by build_index.
    Only returns matches above a minimum relevance score to avoid noise.

    Args:
        question: Natural-language question about the collected papers
        top_k: Number of top matching papers to consider (default 3)
        min_score: Minimum cosine similarity score to include a match (default 0.25)

    Returns:
        Dict with keys:
            - matches: list of paper dicts (title, abstract, arxiv_id, pdf_url, score),
                       filtered to only those above min_score
            - note: present if the index doesn't exist yet, or if no matches clear the threshold
    """
    if not FAISS_PATH.exists() or not META_PATH.exists():
        return {"matches": [], "note": "No index found. Call build_index first."}

    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "rb") as f:
        papers = pickle.load(f)

    embedder = _get_embedder()
    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, min(top_k, len(papers)))

    matches = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < min_score:
            continue
        p = papers[idx]
        matches.append({
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "abstract": p["abstract"],
            "pdf_url": p["pdf_url"],
            "score": float(score),
        })

    if not matches:
        return {"matches": [], "note": "No sufficiently relevant papers found in the collection."}

    return {"matches": matches}

if __name__ == "__main__":
    # Quick standalone test
    build_result = build_index.invoke({"reason": "test run"})
    print("Build index result:", build_result)

    if build_result["indexed_count"] > 0:
        query_result = query_collection.invoke({"question": "papers about video generation", "top_k": 2})
        print("\nQuery result:")
        for m in query_result["matches"]:
            print(f"- [{m['score']:.3f}] {m['title']}")