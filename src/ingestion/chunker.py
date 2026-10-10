"""Simple recursive character text splitter with overlap."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List
import hashlib


@dataclass
class Chunk:
    id: str
    content: str
    source: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _make_id(content: str, source: str, idx: int, metadata: dict | None = None) -> str:
    """Stable unique id: source + page + chunk index + content hash."""
    meta = metadata or {}
    page = meta.get("page", "")
    # Full content hash avoids collisions when pages share headers/footers
    content_hash = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    raw = f"{source}|p{page}|c{idx}|{content_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursive character splitter inspired by LangChain."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining_seps = separators[1:] if len(separators) > 1 else [""]

    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)

    chunks: list[str] = []
    current = ""

    for part in splits:
        candidate = (current + separator + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if len(part) > chunk_size and remaining_seps:
                chunks.extend(
                    split_text(part, chunk_size, chunk_overlap, remaining_seps)
                )
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        prev = chunks[i - 1]
        overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
        overlapped.append((overlap_text + " " + chunk).strip())
    return overlapped


def chunk_document(
    content: str,
    source: str,
    metadata: dict,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Chunk]:
    pieces = split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for i, piece in enumerate(pieces):
        meta = {**metadata, "chunk_index": i}
        chunks.append(
            Chunk(
                id=_make_id(piece, source, i, meta),
                content=piece,
                source=source,
                metadata=meta,
            )
        )
    return chunks
