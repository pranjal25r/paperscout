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
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    reasoning_effort="low",
    max_tokens=4096,
    model_kwargs={"parallel_tool_calls": False},
)

tools = [fetch_arxiv_papers, clean_papers, store_papers, query_stored_papers, build_index, query_collection]

SYSTEM_PROMPT = """When answering questions with query_collection, if it returns no matches (or a note saying
none are relevant), tell the user honestly that the collection doesn't currently have papers
on that topic, rather than stretching unrelated papers to answer the question.
Only when the user asks you to collect or fetch papers: call fetch_arxiv_papers once with the
number they asked for, then call clean_papers and store_papers with the returned batch_id, then
build_index, then report the stored titles.
When the user asks what papers are stored, or asks a question about papers, do NOT fetch from
arXiv. Use query_stored_papers to list papers, or query_collection to answer questions.
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