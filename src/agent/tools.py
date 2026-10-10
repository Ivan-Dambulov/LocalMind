"""Tools the agent can call."""
from __future__ import annotations

from typing import List
from ddgs import DDGS

from src.retrieval.hybrid import HybridRetriever


def local_rag_tool(retriever: HybridRetriever, query: str, top_k: int = 5) -> str:
    hits = retriever.search(query, top_k=top_k)
    if not hits:
        return "No relevant documents found in the local knowledge base."

    parts = []
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", {})
        source = meta.get("file") or meta.get("source", "unknown")
        page = meta.get("page", "")
        cite = f"{source}" + (f", page {page}" if page else "")
        parts.append(f"[{i}] (score={h['score']:.3f}) {cite}\n{h['content']}")
    return "\n\n".join(parts)


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
