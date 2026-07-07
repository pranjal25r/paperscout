"""
Gradio interface for the PaperScout agent.
"""
import gradio as gr
from agent import agent_executor

EXAMPLE_QUERIES = [
    "Collect the 5 most recent papers on retrieval augmented generation",
    "What papers do we have related to AI safety or monitoring?",
    "Collect 3 papers on diffusion transformers",
    "Show me the papers currently stored in the database",
]


def chat_fn(message, history):
    try:
        result = agent_executor.invoke({"input": message})
        return result["output"]
    except Exception as e:
        return f"Something went wrong: {e}\n\nThis is often a temporary rate limit — try again in a moment."


demo = gr.ChatInterface(
    fn=chat_fn,
    title="PaperScout — Autonomous Research Data-Collection Agent",
    description=(
        "A LangChain agent that autonomously collects, cleans, and stores arXiv papers, "
        "then answers questions using retrieval-augmented generation over its own collected data. "
        "Try asking it to collect papers on a topic, or ask a question about what it already knows."
    ),
    examples=EXAMPLE_QUERIES,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()