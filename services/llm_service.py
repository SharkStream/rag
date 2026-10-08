from __future__ import annotations

from typing import Iterable, List

from openai import OpenAI

from config import settings


class LLMService:
    def __init__(
        self,
        api_key: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.llm_model = llm_model or settings.LLM_MODEL
        self.client = (
            OpenAI(api_key=self.api_key, base_url=settings.OPENAI_BASE_URL)
            if self.api_key
            else None
        )

    def get_embeddings(self, texts: Iterable[str]) -> list[list[float]]:
        """Generate embeddings for a list of text chunks."""
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is not configured.")

        text_list = [str(text).strip() for text in texts if str(text).strip()]
        if not text_list:
            return []

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text_list,
        )
        return [item.embedding for item in response.data]

    def generate_answer(self, prompt: str, context: str = "") -> str:
        """Generate a response using the configured LLM model."""
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is not configured.")

        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def generate_response(self, prompt: str, context: str = "") -> str:
        return self.generate_answer(prompt=prompt, context=context)
