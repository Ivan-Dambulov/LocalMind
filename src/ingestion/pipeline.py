"""End-to-end ingestion: load → chunk → (optionally) persist metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.config import settings
from src.ingestion.loader import load_directory, Document
from src.ingestion.chunker import chunk_document, Chunk


def run_ingestion(
    raw_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    raw_dir = raw_dir or settings.raw_dir
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    print(f"[ingestion] Loading documents from {raw_dir}")
    documents: list[Document] = list(load_directory(raw_dir))
    print(f"[ingestion] Loaded {len(documents)} document pages/files")

    all_chunks: list[Chunk] = []
    for doc in tqdm(documents, desc="Chunking"):
        chunks = chunk_document(
            content=doc.content,
            source=doc.source,
            metadata=doc.metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    print(f"[ingestion] Produced {len(all_chunks)} chunks")

    # Persist processed chunks for inspection / re-use
    out_path = settings.processed_dir / "chunks.jsonl"
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"[ingestion] Saved chunks → {out_path}")

    return all_chunks
