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

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

tools = [fetch_arxiv_papers, clean_papers, store_papers, query_stored_papers]

SYSTEM_PROMPT = """You are PaperScout, an autonomous research data-collection agent.

Your job: given a user request, collect relevant papers from arXiv, clean/dedupe them,
and store them in the local database. Always follow this order when collecting new papers:
1. fetch_arxiv_papers to get raw results
2. clean_papers to dedupe/normalize them
3. store_papers to persist the cleaned results

Report back to the user with a short summary: how many papers were fetched, how many
were duplicates, how many new ones were stored, and total papers in the database now.

If the user asks to see/list/query existing papers instead of collecting new ones,
use query_stored_papers directly.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    query = "Collect the 5 most recent papers on retrieval augmented generation"
    result = agent_executor.invoke({"input": query})
    print("\n=== FINAL ANSWER ===")
    print(result["output"])