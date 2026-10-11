"""Lightweight ReAct-style agent with retrieval score validation."""
from __future__ import annotations

from typing import Callable
from dataclasses import dataclass, field

from src.agent.prompts import SYSTEM_PROMPT, TOOL_DECISION_PROMPT, ANSWER_PROMPT
from src.agent.tools import local_rag_tool, web_search_tool
from src.retrieval.hybrid import HybridRetriever
from src.config import settings


REFUSAL_MESSAGE = (
    "I don't have strong enough evidence in the local documents to answer this "
    "confidently (retrieval score below threshold). "
    "Try rephrasing, adding relevant files, or lowering the score threshold in the sidebar."
)


@dataclass
class AgentResult:
    answer: str
    tool_used: str
    context: str
    sources: list[str] = field(default_factory=list)
    # Route validation metadata (for UI / debugging)
    proposed_tool: str = ""
    max_local_score: float | None = None
    route_overridden: bool = False
    override_reason: str = ""


class LocalMindAgent:
    """
    1. Propose a tool (LLM router)
    2. If local retrieval is used, validate scores against threshold
    3. Fallback to web or refuse when local evidence is weak
    4. Generate answer only when context is allowed
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_generate: Callable[[str], str],
    ):
        self.retriever = retriever
        self.llm_generate = llm_generate

    def _decide_tool(self, question: str) -> str:
        prompt = TOOL_DECISION_PROMPT.format(question=question)
        decision = self.llm_generate(prompt).strip().lower()
        for option in ("local_rag", "web_search", "both", "none"):
            if option in decision:
                return option
        return "local_rag"

    def _local_evidence_ok(self, max_score: float) -> bool:
        return max_score >= float(settings.retrieval_min_score)

    def run(self, question: str) -> AgentResult:
        proposed = self._decide_tool(question)
        tool = proposed
        context_parts: list[str] = []
        sources: list[str] = []
        max_local_score: float | None = None
        route_overridden = False
        override_reason = ""

        # --- Local retrieval + score gate ---
        use_local = tool in ("local_rag", "both")
        local_ok = False

        if use_local:
            local_result = local_rag_tool(self.retriever, question)
            max_local_score = local_result.max_score
            local_ok = self._local_evidence_ok(local_result.max_score)

            if local_ok:
                context_parts.append(
                    f"=== LOCAL DOCUMENTS (max_score={local_result.max_score:.3f}) ===\n"
                    + local_result.text
                )
                for line in local_result.text.splitlines():
                    if line.startswith("[") and "]" in line:
                        sources.append(line.split("]")[0] + "]")
            else:
                # Weak local evidence — do not use those chunks for generation
                route_overridden = True
                override_reason = (
                    f"Local max score {local_result.max_score:.3f} "
                    f"< threshold {settings.retrieval_min_score:.3f}"
                )
                context_parts.append(
                    f"=== LOCAL DOCUMENTS (REJECTED: {override_reason}) ===\n"
                    f"Top score too low. Local context not used for generation."
                )

                if tool == "local_rag":
                    action = (settings.weak_local_action or "web").lower()
                    if action == "refuse":
                        return AgentResult(
                            answer=REFUSAL_MESSAGE,
                            tool_used="refused",
                            context="\n\n".join(context_parts),
                            sources=[],
                            proposed_tool=proposed,
                            max_local_score=max_local_score,
                            route_overridden=True,
                            override_reason=override_reason,
                        )
                    # Fallback to web
                    tool = "web_search"
                elif tool == "both":
                    # Keep "both" intent but only web will be added below
                    tool = "web_search"
                    # mark that we intended both but local failed
                    override_reason += " → using web only"

        # --- Web ---
        if tool in ("web_search", "both") or (
            route_overridden and tool == "web_search"
        ):
            web_ctx = web_search_tool(question)
            context_parts.append("=== WEB SEARCH ===\n" + web_ctx)

        if tool == "none":
            context_parts.append("No external context requested.")

        # If we only had rejected local and somehow no web — refuse
        if not context_parts or (
            route_overridden
            and tool == "web_search"
            and all("WEB SEARCH" not in p for p in context_parts)
        ):
            return AgentResult(
                answer=REFUSAL_MESSAGE,
                tool_used="refused",
                context="\n\n".join(context_parts),
                sources=[],
                proposed_tool=proposed,
                max_local_score=max_local_score,
                route_overridden=route_overridden,
                override_reason=override_reason or "No usable context",
            )

        full_context = "\n\n".join(context_parts)
        answer_prompt = ANSWER_PROMPT.format(context=full_context, question=question)
        full_prompt = SYSTEM_PROMPT + "\n\n" + answer_prompt
        answer = self.llm_generate(full_prompt)

        final_tool = tool
        if route_overridden and proposed == "local_rag" and tool == "web_search":
            final_tool = "web_search"
        elif route_overridden and proposed == "both":
            final_tool = "web_search"

        return AgentResult(
            answer=answer,
            tool_used=final_tool,
            context=full_context,
            sources=sources,
            proposed_tool=proposed,
            max_local_score=max_local_score,
            route_overridden=route_overridden,
            override_reason=override_reason,
        )
