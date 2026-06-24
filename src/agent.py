"""
agent.py
--------
Direct RAG chain over retrieved pharma doc chunks.
No AgentExecutor — retrieve → format context → Gemini → answer.

Usage:
    python src/agent.py --query "What are ICH Q1A stability testing requirements?"
"""

import logging
import re
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from retriever import PharmaRetriever
from config import GEMINI_API_KEY, LLM_MODEL, TOP_K

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Pharmaceutical Regulatory Affairs AI Assistant.

Answer questions ONLY using the retrieved document context provided.
ALWAYS cite sources using [Source N: filename, Page X] format inline.
If context is insufficient, say: "The documents provided do not contain enough information."
Never fabricate regulatory requirements, drug names, or thresholds.
Be precise — regulators care about exact wording.

Response format:
- Direct answer first
- Supporting detail with inline citations [Source N]
- Confidence note if context was limited
"""


# ─────────────────────────────────────────────
# RESPONSE SCHEMA
# ─────────────────────────────────────────────

class AgentResponse:
    def __init__(self, answer: str, sources=None, intermediate_steps=None):
        self.answer = answer
        self.sources = sources or []
        self.intermediate_steps = intermediate_steps or []
        self.num_tool_calls = 1  # always 1 retrieval call


# ─────────────────────────────────────────────
# PHARMA QA AGENT (direct RAG, no AgentExecutor)
# ─────────────────────────────────────────────

class PharmaQAAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )
        self.retriever = PharmaRetriever()
        self.chat_history: list = []
        log.info(f"PharmaQAAgent ready — model: {LLM_MODEL}")

    def ask(self, query: str) -> AgentResponse:
        log.info(f"Query: {query}")

        # Step 1 — retrieve relevant chunks
        chunks = self.retriever.retrieve_above_threshold(query, top_k=TOP_K)
        context = self.retriever.format_context(chunks)
        log.info(f"Retrieved {len(chunks)} chunks.")

        # Step 2 — build messages
        user_message = (
            f"Context from regulatory documents:\n\n{context}\n\n"
            f"Question: {query}"
        )

        messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + self.chat_history
            + [HumanMessage(content=user_message)]
        )

        # Step 3 — call Gemini directly
        response = self.llm.invoke(messages)
        answer = response.content

        # Step 4 — extract sources from context labels
        sources = self._extract_sources(context)

        # Step 5 — update chat history (clean query, not context-stuffed)
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))

        return AgentResponse(answer=answer, sources=sources)

    def reset_history(self):
        self.chat_history = []
        log.info("Chat history cleared.")

    def _extract_sources(self, context: str) -> list[dict]:
        sources = []
        matches = re.findall(
            r"\[Source \d+: (.+?), Page (\d+) \| Score: ([\d.]+)\]",
            context
        )
        for source, page, score in matches:
            entry = {"source": source, "page_num": int(page), "score": float(score)}
            if entry not in sources:
                sources.append(entry)
        return sources


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pharma Doc QA — Agent")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    agent = PharmaQAAgent()
    response = agent.ask(args.query)

    print("\n" + "="*60)
    print("ANSWER:")
    print(response.answer)
    print("\nSOURCES:")
    for s in response.sources:
        print(f"  - {s['source']} | Page {s['page_num']} | Score {s['score']}")
