"""Chroma-backed vector store with metadata."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import Embedder


class VectorStore:
    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str = "localmind",
        embedder: Embedder | None = None,
    ):
        self.persist_dir = Path(persist_dir or settings.vectorstore_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection (clean re-ingest)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[vectorstore] Collection '{self.collection_name}' reset")

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        if not chunks:
            return

        # Deduplicate by id (keep first occurrence)
        seen: set[str] = set()
        unique_chunks: list[Chunk] = []
        for c in chunks:
            if c.id in seen:
                continue
            seen.add(c.id)
            unique_chunks.append(c)

        if len(unique_chunks) < len(chunks):
            print(
                f"[vectorstore] Dropped {len(chunks) - len(unique_chunks)} "
                f"duplicate chunk ids"
            )

        texts = [c.content for c in unique_chunks]
        ids = [c.id for c in unique_chunks]
        metadatas = [
            {**c.metadata, "source": c.source, "content_preview": c.content[:120]}
            for c in unique_chunks
        ]

        clean_metas = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metas.append(clean)

        print(f"[vectorstore] Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed(texts)

        for i in range(0, len(unique_chunks), batch_size):
            # upsert = add or update — safe on re-ingest
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size].tolist(),
                documents=texts[i : i + batch_size],
                metadatas=clean_metas[i : i + batch_size],
            )
        print(
            f"[vectorstore] Upserted {len(unique_chunks)} chunks "
            f"into '{self.collection_name}'"
        )

    def similarity_search(
        self,
        query: str,
        top_k: int | None = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        top_k = top_k or settings.top_k
        query_emb = self.embedder.embed_query(query).tolist()

        n = min(top_k, max(self.collection.count(), 1))
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            hits.append(
                {
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
