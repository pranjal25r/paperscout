# PaperScout — Autonomous Research Data-Collection Agent

A LangChain agent that autonomously collects, cleans, deduplicates, and stores arXiv papers, then answers questions about its own collection using retrieval-augmented generation (RAG). Built to explore agentic tool-calling workflows: an LLM decides which tools to invoke and in what order, rather than following a fixed script.

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/pranjal25r/paperscout) <!-- update with your actual Space URL -->

## What it does

Given a natural-language request, the agent autonomously:

1. **Fetches** papers from the arXiv API matching a search query
2. **Cleans** the results — deduplicates by ID and normalized title, strips malformed records
3. **Stores** cleaned records in a local SQLite database, skipping duplicates on re-runs
4. **Indexes** the collection into a FAISS vector store using sentence-transformer embeddings
5. **Answers questions** about the collection using RAG — retrieving only relevant papers above a similarity threshold, then generating a grounded answer

The LLM (not hardcoded logic) decides which of these steps to run and in what sequence, based on the user's request.

## Example interactions

- *"Collect the 5 most recent papers on retrieval augmented generation"* → fetches, cleans, stores, and re-indexes
- *"What papers do we have related to AI safety or monitoring?"* → retrieves relevant stored papers and answers using only their abstracts as evidence
- *"Show me the papers currently stored in the database"* → direct database query, no fetching

## Architecture

```
User request
     │
     ▼
LangChain AgentExecutor (Groq / Llama 3.1 8B)
     │
     ├── fetch_arxiv_papers   → arXiv API
     ├── clean_papers          → dedup + normalize
     ├── store_papers          → SQLite
     ├── query_stored_papers   → SQLite
     ├── build_index           → FAISS + sentence-transformers
     └── query_collection      → FAISS similarity search (with relevance floor)
```

Each tool is a standalone, independently testable Python function wrapped with LangChain's `@tool` decorator — the LLM sees each tool's docstring and decides when and how to call it.

## Tech stack

- **Agent framework:** LangChain (`create_tool_calling_agent`, `AgentExecutor`)
- **LLM:** Groq — `llama-3.1-8b-instant`
- **Retrieval:** FAISS (`IndexFlatIP`, cosine similarity) + `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Storage:** SQLite
- **Data source:** arXiv API
- **UI:** Gradio (`ChatInterface`)
- **Deployment:** Hugging Face Spaces

## Setup

```bash
git clone https://github.com/pranjal25r/paperscout.git
cd paperscout
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Run the agent directly:

```bash
python agent.py
```

Or launch the Gradio UI locally:

```bash
python app.py
```

## Design notes and engineering decisions

**Why `llama-3.1-8b-instant` instead of `llama-3.3-70b-versatile`:**
During development, I benchmarked both models for tool-calling reliability on Groq. Despite 70B's stronger reasoning, it repeatedly failed on structured tool-calling in this workflow — including empty-argument parsing errors, malformed parallel tool calls, and type-mismatched arguments (e.g., generating `"10"` instead of `10` for a numeric parameter). 8B was consistently reliable across the same tasks. Since tool-calling correctness matters more than answer eloquence for an agentic pipeline, I chose 8B for both development and the deployed demo — a deliberate reliability-over-raw-capability tradeoff.

**Relevance filtering in RAG retrieval:**
`query_collection` applies a minimum cosine-similarity threshold (`min_score`, default 0.25) before passing retrieved papers to the LLM. Without this, low-relevance matches (e.g., papers scoring 0.03–0.16 on an unrelated query) got included and diluted the generated answer with off-topic content. The agent is also instructed to say plainly when nothing relevant is found, rather than stretching unrelated papers into an answer.

**Idempotent storage:**
`store_papers` checks for existing `arxiv_id`s before inserting, so re-running a collection request never creates duplicate database entries — verified by running the same insert twice and confirming the second run reports zero new inserts.

**Tool-level testability:**
Every tool (`fetch_arxiv_papers`, `clean_papers`, `store_papers`, `query_stored_papers`, `build_index`, `query_collection`) has a standalone `if __name__ == "__main__"` test block and was verified working in isolation before being wired into the agent — so agent-level bugs could be isolated to orchestration rather than underlying logic.

## Observability

Integrated [LangSmith](https://smith.langchain.com) tracing to inspect agent execution — every LLM call, tool invocation, latency, and token count is automatically captured per run. This surfaced that `query_collection`'s embedding step (via `sentence-transformers`) accounts for the majority of end-to-end latency, not the LLM call itself — useful for future optimization (e.g., persistent embedder caching).

## Known limitations

- No conversation memory between turns — each request is handled independently
- Corpus size is small (demo-scale); FAISS `IndexFlatIP` is exact search and wouldn't scale to large collections without an approximate-nearest-neighbor index
- Relevance filtering (`min_score`) is a fixed threshold, not adaptively tuned per query

## Future improvements

- Add conversational memory so follow-up questions can reference prior turns
- Expand beyond arXiv to additional open data sources
- Add evaluation harness (e.g., LLM-as-judge) to systematically score retrieval quality, similar to [Citera](https://github.com/pranjal25r/Citera)
```