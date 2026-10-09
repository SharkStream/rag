from __future__ import annotations

from typing import Iterable, Protocol

from openai import OpenAI

from config import settings


class EmbeddingBackend(Protocol):
    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        """Return embeddings for the given texts."""


class OpenAIEmbeddingBackend:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or settings.EMBEDDING_MODEL
        self.client = (
            OpenAI(api_key=api_key or settings.OPENAI_API_KEY, base_url=base_url or settings.OPENAI_BASE_URL)
            if api_key or settings.OPENAI_API_KEY
            else None
        )

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        """Generate embeddings using the OpenAI embedding API."""
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is not configured for embeddings.")

        clean_texts = [str(text).strip() for text in texts if str(text).strip()]
        if not clean_texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=clean_texts,
        )
        return [item.embedding for item in response.data]


class DashScopeEmbeddingBackend:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.DASHSCOPE_API_KEY
        self.model = model or settings.DASHSCOPE_EMBEDDING_MODEL

        if not self.api_key:
            self.client = None
            return

        import dashscope

        dashscope.api_key = self.api_key
        self.client = dashscope

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        """Generate embeddings using the DashScope embedding API."""
        if self.client is None:
            raise ValueError("DASHSCOPE_API_KEY is not configured for embeddings.")

        clean_texts = [str(text).strip() for text in texts if str(text).strip()]
        if not clean_texts:
            return []

        payload = clean_texts[0] if len(clean_texts) == 1 else clean_texts
        response = self.client.TextEmbedding.call(model=self.model, input=payload)

        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        if output is None:
            raise ValueError(f"Unexpected DashScope embedding response: {response}")

        embeddings = output.get("embeddings", []) if isinstance(output, dict) else []
        if not embeddings:
            raise ValueError(f"No embeddings returned from DashScope: {response}")

        if isinstance(embeddings[0], dict) and "embedding" in embeddings[0]:
            return [item["embedding"] for item in embeddings]
        if isinstance(embeddings[0], list):
            return embeddings

        raise ValueError(f"Unsupported DashScope embedding response format: {response}")


class SentenceTransformersEmbeddingBackend:
    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        self.model_name = model or settings.LOCAL_EMBEDDING_MODEL
        self.device = device or settings.LOCAL_EMBEDDING_DEVICE

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        """Generate embeddings locally with sentence-transformers."""
        clean_texts = [str(text).strip() for text in texts if str(text).strip()]
        if not clean_texts:
            return []

        embeddings = self.model.encode(clean_texts, convert_to_numpy=True, normalize_embeddings=False)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(vector) for vector in embeddings]
