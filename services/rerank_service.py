from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from config import settings


class RerankEngine:
    """Re-rank retrieved documents with a CrossEncoder model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.RERANK_MODEL
        self.device = device or "cpu"
        self.model = CrossEncoder(self.model_name, device=self.device)

    def _extract_document_text(self, document: dict[str, Any]) -> str:
        payload = document.get("payload") or {}
        return payload.get("text") or payload.get("content") or ""

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Reorder retrieved documents by cross-encoder relevance score."""
        if not query or not documents:
            return []

        valid_documents: list[dict[str, Any]] = []
        pairs: list[list[str]] = []

        for document in documents:
            text = self._extract_document_text(document)
            if not text:
                continue
            valid_documents.append(document)
            pairs.append([query, text])

        if not pairs:
            return []

        scores = self.model.predict(pairs)

        ranked_documents: list[dict[str, Any]] = []
        for document, score in zip(valid_documents, scores):
            ranked_document = dict(document)
            ranked_document["rerank_score"] = float(score)
            ranked_documents.append(ranked_document)

        ranked_documents.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
        return ranked_documents[:top_k]
