from __future__ import annotations

from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import settings


class QdrantService:
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.vector_size = vector_size or settings.QDRANT_VECTOR_SIZE
        self.client = QdrantClient(url=self.url, api_key=self.api_key or None)

    def ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists."""
        collections = [item.name for item in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_documents(self, documents: Iterable[dict[str, Any]]) -> None:
        """Insert or update vectors in Qdrant."""
        self.ensure_collection()
        points = []
        for item in documents:
            points.append(
                models.PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload=item.get("payload", {}),
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(self, query_vector: list[float], limit: int = 5, score_threshold: float | None = None) -> list[dict[str, Any]]:
        """Search for similar documents."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        return [
            {"id": item.id, "score": item.score, "payload": item.payload}
            for item in results
        ]

    def list_documents(self, limit: int = 100, source: str | None = None) -> list[dict[str, Any]]:
        """List stored documents with their payload metadata, optionally filtered by source."""
        self.ensure_collection()
        scroll_filter = None
        if source:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source),
                    )
                ]
            )

        response = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            filter=scroll_filter,
        )
        points = response[0]
        return [
            {"id": point.id, "payload": point.payload}
            for point in points
        ]

    def delete_documents_by_source(self, source: str) -> int:
        """Delete all documents belonging to the provided source name."""
        self.ensure_collection()
        source_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source),
                )
            ]
        )
        points = self.list_documents(source=source)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=source_filter),
        )
        return len(points)

    def clear_collection(self) -> None:
        """Delete and recreate the collection to clear all stored documents."""
        self.delete_collection()
        self.ensure_collection()

    def delete_collection(self) -> None:
        """Delete the configured collection."""
        if self.collection_name in [item.name for item in self.client.get_collections().collections]:
            self.client.delete_collection(self.collection_name)
