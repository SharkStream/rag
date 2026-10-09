from services.llm_service import LLMService


class StubEmbeddingBackend:
    def __init__(self):
        self.calls = []

    def get_embeddings(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


def test_custom_embedding_backend_does_not_require_openai_key(monkeypatch):
    monkeypatch.setattr("config.settings.OPENAI_API_KEY", "")

    backend = StubEmbeddingBackend()
    service = LLMService(
        api_key="",
        llm_model="local-model",
        embedding_backend=backend,
    )

    embeddings = service.get_embeddings(["hello", "world"])

    assert embeddings == [[1.0, 0.0], [1.0, 0.0]]
    assert backend.calls == [["hello", "world"]]


if __name__ == "__main__":
    import pathlib

    backend = StubEmbeddingBackend()
    service = LLMService(
        api_key="",
        llm_model="local-model",
        embedding_backend=backend,
    )
    result = service.get_embeddings(["hello", "world"])
    print(result)
    print("OK")
