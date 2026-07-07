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