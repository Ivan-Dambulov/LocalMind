"""Tools the agent can call."""
from __future__ import annotations

from dataclasses import dataclass, field
from ddgs import DDGS

from src.retrieval.hybrid import HybridRetriever
from src.config import settings


@dataclass
class LocalRagResult:
    text: str
    hits: list[dict] = field(default_factory=list)
    max_score: float = 0.0
    mean_top3: float = 0.0


def local_rag_tool(
    retriever: HybridRetriever,
    query: str,
    top_k: int | None = None,
) -> LocalRagResult:
    top_k = top_k or settings.top_k
    hits = retriever.search(query, top_k=top_k)
    if not hits:
        return LocalRagResult(
            text="No relevant documents found in the local knowledge base.",
            hits=[],
            max_score=0.0,
            mean_top3=0.0,
        )

    scores = [float(h.get("score", 0.0)) for h in hits]
    max_score = max(scores) if scores else 0.0
    top3 = scores[:3]
    mean_top3 = sum(top3) / len(top3) if top3 else 0.0

    parts = []
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", {})
        source = meta.get("file") or meta.get("source", "unknown")
        page = meta.get("page", "")
        cite = f"{source}" + (f", page {page}" if page else "")
        parts.append(f"[{i}] (score={h['score']:.3f}) {cite}\n{h['content']}")

    return LocalRagResult(
        text="\n\n".join(parts),
        hits=hits,
        max_score=max_score,
        mean_top3=mean_top3,
    )


def web_search_tool(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No web results found."
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            parts.append(f"[{i}] {title}\n{body}\nURL: {href}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Web search failed: {e}"
