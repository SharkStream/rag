from __future__ import annotations

from pathlib import Path
from typing import Any

from config import settings

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


def _read_text_bytes(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md", ".csv", ".json", ".log"}:
        return data.decode("utf-8", errors="replace")

    if suffix == ".pdf" and PdfReader is not None:
        text_parts: list[str] = []
        try:
            import io

            pdf_file = io.BytesIO(data)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception:
            return ""

    return data.decode("utf-8", errors="replace")


def extract_text_from_file(file_obj: Any, filename: str | None = None) -> str:
    """Extract text content from a file-like object or uploaded file."""
    name = filename or getattr(file_obj, "name", "uploaded_file")
    data = file_obj.read() if hasattr(file_obj, "read") else file_obj
    return _read_text_bytes(data if isinstance(data, bytes) else bytes(data), name)


def index_text_to_qdrant(
    text: str,
    source_name: str,
    llm_service,
    qdrant_service,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Split text, generate embeddings, and insert them into Qdrant."""
    if not text or not text.strip():
        return 0

    from uuid import uuid4

    from utils.text_splitter import split_text_by_chunks

    chunk_size = chunk_size if chunk_size is not None else settings.DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.DEFAULT_CHUNK_OVERLAP

    chunks = split_text_by_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return 0

    documents = []
    for index, chunk in enumerate(chunks):
        vector = llm_service.get_embeddings([chunk])[0]
        documents.append(
            {
                "id": str(uuid4()),
                "vector": vector,
                "payload": {
                    "text": chunk,
                    "source": source_name,
                    "chunk_index": index,
                },
            }
        )

    qdrant_service.upsert_documents(documents)
    return len(documents)
