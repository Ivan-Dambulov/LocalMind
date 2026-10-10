"""Hybrid retrieval: dense (Chroma) + BM25 keyword search with score fusion."""
from __future__ import annotations

from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

from src.config import settings
from src.retrieval.vectorstore import VectorStore
from src.ingestion.chunker import Chunk


class HybridRetriever:
    def __init__(self, vectorstore: VectorStore, chunks: List[Chunk] | None = None):
        self.vs = vectorstore
        self.chunks = chunks or []
        self._bm25 = None
        self._corpus_tokens = None
        self._id_to_chunk = {c.id: c for c in self.chunks}

        if self.chunks:
            self._build_bm25()

    def _build_bm25(self) -> None:
        self._corpus_tokens = [c.content.lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        print(f"[hybrid] BM25 index built over {len(self.chunks)} chunks")

    def set_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self._id_to_chunk = {c.id: c for c in chunks}
        self._build_bm25()

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        hits = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            hits.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": {**chunk.metadata, "source": chunk.source},
                    "score": float(scores[idx]),
                    "source_type": "bm25",
                }
            )
        return hits

    def search(
        self,
        query: str,
        top_k: int | None = None,
        alpha: float | None = None,
    ) -> list[dict]:
        """
        alpha: weight for dense score (1.0 = pure dense, 0.0 = pure BM25)
        """
        top_k = top_k or settings.top_k
        alpha = alpha if alpha is not None else settings.hybrid_alpha

        dense_hits = self.vs.similarity_search(query, top_k=top_k * 2)
        bm25_hits = self._bm25_search(query, top_k=top_k * 2)

        # Normalize scores to [0, 1]
        def normalize(hits: list[dict]) -> dict[str, float]:
            if not hits:
                return {}
            scores = [h["score"] for h in hits]
            mn, mx = min(scores), max(scores)
            if mx - mn < 1e-9:
                return {h["id"]: 1.0 for h in hits}
            return {h["id"]: (h["score"] - mn) / (mx - mn) for h in hits}

        dense_norm = normalize(dense_hits)
        bm25_norm = normalize(bm25_hits)

        # Fuse
        all_ids = set(dense_norm) | set(bm25_norm)
        fused: list[dict] = []
        id_to_hit = {h["id"]: h for h in dense_hits + bm25_hits}

        for doc_id in all_ids:
            d = dense_norm.get(doc_id, 0.0)
            b = bm25_norm.get(doc_id, 0.0)
            score = alpha * d + (1 - alpha) * b
            base = id_to_hit[doc_id]
            fused.append(
                {
                    "id": doc_id,
                    "content": base["content"],
                    "metadata": base["metadata"],
                    "score": score,
                    "dense_score": d,
                    "bm25_score": b,
                }
            )

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]
