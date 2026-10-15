from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from config import settings


class RedisCache:
    """Query-level cache for RAG retrieval results using Redis."""

    def __init__(self, url: str | None = None, ttl_seconds: int | None = None) -> None:
        self.client = redis.Redis.from_url(url or settings.REDIS_URL, decode_responses=True)
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.REDIS_CACHE_TTL

    def _build_key(self, query: str) -> str:
        normalized = query.strip().lower()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"rag:query:{digest}"

    def get(self, query: str) -> list[dict[str, Any]] | None:
        key = self._build_key(query)
        raw = self.client.get(key)
        if raw is None:
            return None

        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                return payload
        except (TypeError, ValueError):
            return None

        return None

    def set(self, query: str, value: list[dict[str, Any]]) -> None:
        key = self._build_key(query)
        payload = json.dumps(value, ensure_ascii=False)
        self.client.setex(key, self.ttl_seconds, payload)

    def delete(self, query: str) -> None:
        key = self._build_key(query)
        self.client.delete(key)

    def clear(self) -> None:
        for key in self.client.scan_iter(match="rag:query:*"):
            self.client.delete(key)
