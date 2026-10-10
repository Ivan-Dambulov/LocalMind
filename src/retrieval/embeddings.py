"""Embedding model wrapper."""
from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        print(f"[embeddings] Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-friendly
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
