"""
PaperScout Agent: LangChain agent that orchestrates arXiv fetch, clean, and storage tools.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from tools_arxiv import fetch_arxiv_papers
from tools_clean import clean_papers
from tools_storage import store_papers, query_stored_papers
from tools_rag import build_index, query_collection


load_dotenv()

llm = ChatGroq(
    # model="llama-3.3-70b-versatile",
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    model_kwargs={"parallel_tool_calls": False},
)

tools = [fetch_arxiv_papers, clean_papers, store_papers, query_stored_papers, build_index, query_collection]

SYSTEM_PROMPT = """You are PaperScout, an autonomous research data-collection agent.

Your job: given a user request, collect relevant papers from arXiv, clean/dedupe them,
and store them in the local database. Always follow this order when collecting new papers:
1. fetch_arxiv_papers to get raw results
2. clean_papers to dedupe/normalize them
3. store_papers to persist the cleaned results
4. build_index to refresh the searchable index with any newly stored papers

Report back to the user with a short summary: how many papers were fetched, how many
were duplicates, how many new ones were stored, and total papers in the database now.

If the user asks to see/list existing papers, use query_stored_papers directly.

If the user asks a QUESTION about the collected papers (e.g. "what papers do we have
on X", "summarize what we know about Y"), use query_collection to retrieve the most
relevant stored papers, then answer the question yourself using only the retrieved
abstracts as evidence. If query_collection returns a note saying no index exists yet,
call build_index first, then retry query_collection.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


import time

def run_with_retry(executor, query, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return executor.invoke({"input": query})
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"Retrying after error: {e}")
            time.sleep(2)

# if __name__ == "__main__":
#     result1 = run_with_retry(agent_executor, "Collect the 5 most recent papers on large language model safety")
#     print("\n=== COLLECT RESULT ===")
#     print(result1["output"])

#     result2 = run_with_retry(agent_executor, "What papers do we have related to AI safety or monitoring?")
#     print("\n=== QUERY RESULT ===")
#     print(result2["output"])

import time

if __name__ == "__main__":
    result1 = agent_executor.invoke({"input": "Collect the 2 most recent papers on large language model safety"})
    print("\n=== COLLECT RESULT ===")
    print(result1["output"])

    time.sleep(15)  # let the per-minute token budget reset

    result2 = agent_executor.invoke({"input": "What papers do we have related to AI safety or monitoring?"})
    print("\n=== QUERY RESULT ===")
    print(result2["output"])