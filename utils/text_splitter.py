from __future__ import annotations

import re
from typing import List


def split_text_by_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks.

    This helper keeps the implementation lightweight and avoids extra dependencies.
    """
    if not text or chunk_size <= 0:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            split_index = chunk.rfind(" ")
            if split_index > 0:
                chunk = chunk[:split_index]
                end = start + split_index
        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(0, end - chunk_overlap)

    return [chunk for chunk in chunks if chunk]


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence split by punctuation."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
