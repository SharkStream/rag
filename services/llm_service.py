from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from config import settings
from prompts.service import PromptService
from services.embeddings import (
    EmbeddingBackend,
    OpenAIEmbeddingBackend,
    DashScopeEmbeddingBackend,
    SentenceTransformersEmbeddingBackend,
)


class LLMService:
    def __init__(
        self,
        api_key: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.llm_model = llm_model or settings.LLM_MODEL
        self.client = (
            OpenAI(api_key=self.api_key, base_url=settings.OPENAI_BASE_URL)
            if self.api_key
            else None
        )
        self.embedding_backend = embedding_backend or self._build_embedding_backend()
        self.prompt_service = PromptService()

    def _build_embedding_backend(self) -> EmbeddingBackend:
        provider = settings.EMBEDDING_PROVIDER.lower()
        if provider in {"dashscope", "qwen", "ali"}:
            return DashScopeEmbeddingBackend(
                api_key=settings.DASHSCOPE_API_KEY,
                model=self.embedding_model,
            )
        if provider in {"local", "sentence_transformers", "sentence-transformers"}:
            return SentenceTransformersEmbeddingBackend(
                model=settings.LOCAL_EMBEDDING_MODEL,
                device=settings.LOCAL_EMBEDDING_DEVICE,
            )

        return OpenAIEmbeddingBackend(
            api_key=self.api_key,
            model=self.embedding_model,
            base_url=settings.OPENAI_BASE_URL,
        )

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        """Generate embeddings for a list of text chunks."""
        return self.embedding_backend.get_embeddings(texts)

    def _build_messages(
        self,
        prompt: str,
        context: str = "",
        language: str = "Chinese",
        prompt_name: str | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages using the prompt template registry."""
        system_prompt = self.prompt_service.render(
            "chat.system",
            system_role="You are a helpful assistant.",
            language=language,
        )

        effective_prompt_name = prompt_name or (
            "rag.answer_with_context" if context else "rag.answer_without_context"
        )

        if context:
            user_prompt = self.prompt_service.render(
                effective_prompt_name,
                question=prompt,
                context=context,
                language=language,
            )
        else:
            user_prompt = self.prompt_service.render(
                effective_prompt_name,
                question=prompt,
                language=language,
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def rewrite_query(self, question: str, language: str = "Chinese") -> str:
        """Rewrite a user question using a dedicated prompt template."""
        return self.prompt_service.render(
            "rag.rewrite_query",
            question=question,
            language=language,
        )

    def generate_answer(
        self,
        prompt: str,
        context: str = "",
        *,
        prompt_name: str | None = None,
        language: str = "Chinese",
    ) -> str:
        """Generate a response using the configured LLM model and template-based prompts."""
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is not configured.")

        messages = self._build_messages(
            prompt=prompt,
            context=context,
            language=language,
            prompt_name=prompt_name,
        )

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def generate_response(self, prompt: str, context: str = "") -> str:
        return self.generate_answer(prompt=prompt, context=context)
