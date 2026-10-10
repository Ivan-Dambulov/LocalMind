#!/usr/bin/env python3
"""Run the full ingestion pipeline and build the vector store + BM25 index."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.pipeline import run_ingestion
from src.retrieval.embeddings import Embedder
from src.retrieval.vectorstore import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.config import settings


def main():
    print("=" * 60)
    print("LocalMind – Document Ingestion")
    print("=" * 60)

    chunks = run_ingestion()

    if not chunks:
        print("No chunks produced. Add documents to data/raw/ and try again.")
        return

    embedder = Embedder()
    vs = VectorStore(embedder=embedder)

    # Clean re-ingest so old/duplicate ids never conflict
    vs.reset()
    vs.add_chunks(chunks)

    retriever = HybridRetriever(vectorstore=vs, chunks=chunks)
    print(f"\n[done] Vector store now contains {vs.count()} items")
    print(f"Persist path: {settings.vectorstore_dir}")
    print("You can now run: streamlit run src/ui/app.py")


if __name__ == "__main__":
    main()
