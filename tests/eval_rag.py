from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eval_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            dataset.append(json.loads(line))
    return dataset


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    hits = set(top_k) & set(relevant)
    return len(hits) / len(relevant)


def average_precision(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / idx

    return precision_sum / len(relevant)


def evaluate_dataset(
    dataset: list[dict[str, Any]],
    retrieved_by_query: dict[str, list[str]],
    k: int = 5,
) -> dict[str, float]:
    recall_scores: list[float] = []
    ap_scores: list[float] = []

    for item in dataset:
        query_id = item["id"]
        relevant = item["ground_truth"]["positive"]
        retrieved = retrieved_by_query.get(query_id, [])

        recall_scores.append(recall_at_k(retrieved, relevant, k=k))
        ap_scores.append(average_precision(retrieved, relevant))

    return {
        "recall_at_5": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        "mAP": sum(ap_scores) / len(ap_scores) if ap_scores else 0.0,
    }
