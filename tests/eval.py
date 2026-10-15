from __future__ import annotations

from pathlib import Path

from services.llm_service import LLMService
from services.qdrant_service import QdrantService
from services.rerank_service import RerankEngine
from config import settings
from tests.eval_rag import evaluate_dataset, load_eval_dataset


llm_service = LLMService()
qdrant_service = QdrantService()
rerank_engine = RerankEngine()


def retrieve_docs_for_query(query: str, top_k: int = 5) -> list[str]:
    query_vector = llm_service.get_embeddings([query])[0]

    candidate_hits = qdrant_service.search(
        query_vector=query_vector,
        limit=settings.RETRIEVAL_LIMIT,
        score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD,
    )

    reranked_hits = rerank_engine.rerank(
        query,
        candidate_hits,
        top_k=top_k,
    )

    return [
        hit.get("id")
        for hit in reranked_hits
        if hit.get("id") is not None
    ]


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset_path = root / "tests" / "data" / "eval_dataset.jsonl"
    dataset = load_eval_dataset(dataset_path)

    retrieved_by_query = {}
    for item in dataset:
        query_id = item["id"]
        query_text = item["query"]
        retrieved_by_query[query_id] = retrieve_docs_for_query(query_text, top_k=5)

    metrics = evaluate_dataset(dataset, retrieved_by_query, k=5)
    print(metrics)


if __name__ == "__main__":
    main()
