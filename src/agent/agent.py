"""Lightweight ReAct-style agent for LocalMind."""
from __future__ import annotations

from typing import Optional, Callable
from dataclasses import dataclass

from src.agent.prompts import SYSTEM_PROMPT, TOOL_DECISION_PROMPT, ANSWER_PROMPT
from src.agent.tools import local_rag_tool, web_search_tool
from src.retrieval.hybrid import HybridRetriever
from src.config import settings


@dataclass
class AgentResult:
    answer: str
    tool_used: str
    context: str
    sources: list[str]


class LocalMindAgent:
    """
    Simple but effective agent:
    1. Decide which tool(s) to use
    2. Call the tool(s)
    3. Generate final answer with citations
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_generate: Callable[[str], str],
    ):
        self.retriever = retriever
        self.llm_generate = llm_generate  # function that takes prompt → string

    def _decide_tool(self, question: str) -> str:
        prompt = TOOL_DECISION_PROMPT.format(question=question)
        decision = self.llm_generate(prompt).strip().lower()
        # Hard sanitize
        for option in ("local_rag", "web_search", "both", "none"):
            if option in decision:
                return option
        return "local_rag"  # safe default

    def run(self, question: str) -> AgentResult:
        tool = self._decide_tool(question)
        context_parts = []
        sources = []

        if tool in ("local_rag", "both"):
            local_ctx = local_rag_tool(self.retriever, question)
            context_parts.append("=== LOCAL DOCUMENTS ===\n" + local_ctx)
            # crude source extraction
            for line in local_ctx.splitlines():
                if line.startswith("[") and "]" in line:
                    sources.append(line.split("]")[0] + "]")

        if tool in ("web_search", "both"):
            web_ctx = web_search_tool(question)
            context_parts.append("=== WEB SEARCH ===\n" + web_ctx)

        if tool == "none":
            context_parts.append("No external context requested.")

        full_context = "\n\n".join(context_parts)
        answer_prompt = ANSWER_PROMPT.format(context=full_context, question=question)
        # Prepend system style instruction
        full_prompt = SYSTEM_PROMPT + "\n\n" + answer_prompt
        answer = self.llm_generate(full_prompt)

        return AgentResult(
            answer=answer,
            tool_used=tool,
            context=full_context,
            sources=sources,
        )
