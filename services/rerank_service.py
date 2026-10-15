from __future__ import annotations

from typing import Any

from dashscope import TextReRank
from sentence_transformers import CrossEncoder

from config import settings


class RerankEngine:
    """Re-rank retrieved documents using either a local cross-encoder or the DashScope Qwen rerank API."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.RERANK_MODEL
        self.device = device or "cpu"
        self.model_selector = {
            settings.RERANK_MODEL_TYPES.DASH_SCOPE: self._dashscope,
            settings.RERANK_MODEL_TYPES.CROSS_ENCODER: self._cross_encoder,
        }
        self.model_func = self.model_selector.get(
            settings.RERANK_MODEL_MAPPING.get(self.model_name)
        )
        if settings.RERANK_MODEL_MAPPING.get(self.model_name) == "cross_encoder":
            self.model = CrossEncoder(self.model_name, device=self.device)

    def _extract_document_text(self, document: dict[str, Any]) -> str:
        payload = document.get("payload") or {}
        return payload.get("text") or payload.get("content") or ""

    def _call_dashscope_rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        api_key = settings.DASHSCOPE_API_KEY
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not configured for rerank.")

        response = TextReRank.call(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=max(1, top_k),
            return_documents=True,
        )

        if response.status_code != 200:
            raise ValueError(
                f"DashScope rerank failed: code={response.code}, message={response.message}"
            )

        ranked: list[tuple[int, float]] = []
        for item in getattr(response.output, "results", []) or []:
            index = getattr(item, "index", None)
            score = getattr(item, "relevance_score", getattr(item, "score", 0.0))
            if index is None:
                continue
            ranked.append((int(index), float(score)))
        return ranked

    def _dashscope(self, query: str, valid_documents: list[dict[str, Any]], texts: list[str], top_k: int) -> list[tuple[int, float]]:
        ranked_pairs = self._call_dashscope_rerank(query, texts, top_k)
        if not ranked_pairs:
            return []
        
        ranked_by_index = {index: score for index, score in ranked_pairs}
        ranked_documents: list[dict[str, Any]] = []
        for index, document in enumerate(valid_documents):
            if index not in ranked_by_index:
                continue
            ranked_document = dict(document)
            ranked_document["rerank_score"] = float(ranked_by_index[index])
            ranked_documents.append(ranked_document)
        
        ranked_documents.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
        return ranked_documents[:top_k]
    
    def _cross_encoder(self, query: str, valid_documents: list[dict[str, Any]], texts: list[str], top_k: int) -> list[tuple[int, float]]:
        pairs: list[list[str]] = [[query, text] for text in texts]
        scores = self.model.predict(pairs)
    
        ranked_documents: list[dict[str, Any]] = []
        for document, score in zip(valid_documents, scores):
            ranked_document = dict(document)
            ranked_document["rerank_score"] = float(score)
            ranked_documents.append(ranked_document)
    
        ranked_documents.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
        return ranked_documents[:top_k]
    
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Reorder retrieved documents by relevance score."""
        if not query or not documents:
            return []

        valid_documents: list[dict[str, Any]] = []
        texts: list[str] = []

        for document in documents:
            text = self._extract_document_text(document)
            if not text:
                continue
            valid_documents.append(document)
            texts.append(text)

        if not valid_documents:
            return []

        return self.model_func(query, valid_documents, texts, top_k)
            

        
