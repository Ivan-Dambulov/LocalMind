"""Ingestion service used by CLI and UI."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.config import settings
from src.ingestion.pipeline import run_ingestion
from src.retrieval.embeddings import Embedder
from src.retrieval.vectorstore import VectorStore
from src.ingestion.chunker import Chunk


def save_uploads(files, raw_dir: Path | None = None) -> list[str]:
    """Save Streamlit UploadedFile objects into data/raw/. Returns saved names."""
    raw_dir = Path(raw_dir or settings.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.name).name
        dest = raw_dir / name
        dest.write_bytes(f.getvalue())
        saved.append(name)
    return saved


def list_raw_documents(raw_dir: Path | None = None) -> list[dict]:
    raw_dir = Path(raw_dir or settings.raw_dir)
    if not raw_dir.exists():
        return []
    out = []
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".markdown"}:
            out.append({"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1), "path": str(p)})
    return out


def delete_raw_document(name: str, raw_dir: Path | None = None) -> bool:
    raw_dir = Path(raw_dir or settings.raw_dir)
    path = raw_dir / Path(name).name
    if path.exists() and path.is_file():
        path.unlink()
        return True
    return False


def run_full_ingest(
    progress: Callable[[str], None] | None = None,
) -> dict:
    """
    Full ingest: chunk → reset vector store → embed → upsert.
    Returns stats dict.
    """
    def log(msg: str):
        if progress:
            progress(msg)

    log("Loading and chunking documents…")
    chunks = run_ingestion()
    if not chunks:
        return {"ok": False, "chunks": 0, "message": "No documents found in data/raw/"}

    log(f"Embedding {len(chunks)} chunks…")
    embedder = Embedder()
    vs = VectorStore(embedder=embedder)
    log("Resetting vector store…")
    vs.reset()
    log("Upserting into Chroma…")
    vs.add_chunks(chunks)

    # Persist chunks path already done in pipeline
    return {
        "ok": True,
        "chunks": len(chunks),
        "stored": vs.count(),
        "message": f"Indexed {vs.count()} chunks successfully",
    }


def load_chunks_from_disk() -> list[Chunk]:
    import json
    chunks_path = settings.processed_dir / "chunks.jsonl"
    chunks: list[Chunk] = []
    if not chunks_path.exists():
        return chunks
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(
                Chunk(
                    id=d["id"],
                    content=d["content"],
                    source=d["source"],
                    metadata=d["metadata"],
                )
            )
    return chunks
