# src/rag/vectorstore.py
from __future__ import annotations

import logging
import os
from typing import List, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.rag.embeddings import embed_documents, embed_query

logger = logging.getLogger(__name__)

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX_NAME = "socratic_docs"
VECTOR_DIMS = 1024  # bge-m3 dense vector dimension

_client: Elasticsearch | None = None

INDEX_MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "korean": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "korean"},
            "dense_vector": {
                "type": "dense_vector",
                "dims": VECTOR_DIMS,
                "index": True,
                "similarity": "cosine",
            },
            "source": {"type": "keyword"},
            "page": {"type": "integer"},
            "chunk_index": {"type": "integer"},
        }
    },
}


def get_client() -> Elasticsearch:
    global _client
    if _client is None:
        _client = Elasticsearch(ES_URL)
    return _client


def ensure_index() -> None:
    client = get_client()
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
        logger.info(f"Created ES index: {INDEX_NAME}")


def add_documents(docs: List[str], metadatas: List[dict] | None = None, ids: List[str] | None = None) -> int:
    ensure_index()
    client = get_client()

    if ids is None:
        import uuid
        ids = [str(uuid.uuid4()) for _ in docs]

    if metadatas is None:
        metadatas = [{}] * len(docs)

    vectors = embed_documents(docs)

    actions = [
        {
            "_op_type": "create",  # 이미 존재하면 스킵
            "_index": INDEX_NAME,
            "_id": id_,
            "_source": {
                "text": doc,
                "dense_vector": vector,
                **meta,
            },
        }
        for doc, meta, id_, vector in zip(docs, metadatas, ids, vectors)
    ]

    # Wait for refresh so newly uploaded docs are immediately visible to listing APIs.
    success, _ = bulk(client, actions, raise_on_error=False, refresh="wait_for")
    return success


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """여러 검색 결과 리스트를 RRF로 합산합니다. score = Σ 1/(k + rank)"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


SIMILARITY_THRESHOLD = float(os.getenv("SOCRAITES_SIMILARITY_THRESHOLD", "0.4"))


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    import math
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def query(query_text: str, k: int = 5, selected_sources: list[str] | None = None) -> list[dict[str, Any]]:
    ensure_index()
    client = get_client()

    query_vector = embed_query(query_text)
    window = k * 4

    # BM25 검색 (text, dense_vector, source, page 모두 추출)
    bm25_query = {"match": {"text": {"query": query_text}}}
    if selected_sources:
        bm25_query = {
            "bool": {
                "must": bm25_query,
                "filter": {"terms": {"source": selected_sources}}
            }
        }

    bm25_resp = client.search(
        index=INDEX_NAME,
        body={"query": bm25_query, "size": window},
    )
    bm25_hits = {}
    for h in bm25_resp["hits"]["hits"]:
        bm25_hits[h["_id"]] = {
            "text": h["_source"]["text"],
            "dense_vector": h["_source"].get("dense_vector"),
            "source": h["_source"].get("source", "Unknown"),
            "page": h["_source"].get("page", 1),
        }
    bm25_ranking = list(bm25_hits.keys())

    # Dense KNN 검색 (text, dense_vector, source, page 모두 추출)
    knn_body = {"field": "dense_vector", "query_vector": query_vector, "k": window, "num_candidates": window * 2}
    if selected_sources:
        knn_body["filter"] = {"terms": {"source": selected_sources}}

    knn_resp = client.search(
        index=INDEX_NAME,
        body={"knn": knn_body, "size": window},
    )
    knn_hits = {}
    for h in knn_resp["hits"]["hits"]:
        knn_hits[h["_id"]] = {
            "text": h["_source"]["text"],
            "dense_vector": h["_source"].get("dense_vector"),
            "source": h["_source"].get("source", "Unknown"),
            "page": h["_source"].get("page", 1),
        }
    knn_ranking = list(knn_hits.keys())

    # RRF 합산 대상 중 코사인 유사도 필터링 수행
    all_docs = {**bm25_hits, **knn_hits}
    
    filtered_bm25_ranking = []
    filtered_knn_ranking = []
    
    for doc_id in bm25_ranking:
        doc_data = all_docs.get(doc_id)
        if doc_data and doc_data["dense_vector"]:
            sim = _cosine_similarity(query_vector, doc_data["dense_vector"])
            if sim >= SIMILARITY_THRESHOLD:
                filtered_bm25_ranking.append(doc_id)
            else:
                logger.debug(f"Filtering out BM25 hit {doc_id} due to low similarity: {sim:.4f}")
                
    for doc_id in knn_ranking:
        doc_data = all_docs.get(doc_id)
        if doc_data and doc_data["dense_vector"]:
            sim = _cosine_similarity(query_vector, doc_data["dense_vector"])
            if sim >= SIMILARITY_THRESHOLD:
                filtered_knn_ranking.append(doc_id)
            else:
                logger.debug(f"Filtering out KNN hit {doc_id} due to low similarity: {sim:.4f}")

    # 필터링된 결과로 RRF 합산
    rrf_scores = _rrf([filtered_bm25_ranking, filtered_knn_ranking])
    top_k = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    return [
        {
            "text": all_docs[doc_id]["text"],
            "score": score,
            "metadata": {
                "source": all_docs[doc_id]["source"],
                "page": all_docs[doc_id]["page"],
            }
        }
        for doc_id, score in top_k if doc_id in all_docs
    ]


def get_registered_documents() -> List[str]:
    client = get_client()
    if not client.indices.exists(index=INDEX_NAME):
        return []

    body = {
        "size": 0,
        "aggs": {
            "unique_sources": {
                "terms": {
                    "field": "source",
                    "size": 10000
                }
            }
        }
    }
    try:
        resp = client.search(index=INDEX_NAME, body=body)
        buckets = resp.get("aggregations", {}).get("unique_sources", {}).get("buckets", [])
        return [bucket["key"] for bucket in buckets]
    except Exception as e:
        logger.error(f"Failed to get registered documents: {e}")
        return []


def delete_document(source_name: str) -> int:
    client = get_client()
    if not client.indices.exists(index=INDEX_NAME):
        return 0

    body = {
        "query": {
            "term": {
                "source": source_name
            }
        }
    }
    try:
        resp = client.delete_by_query(index=INDEX_NAME, body=body, refresh=True)
        deleted = resp.get("deleted", 0)
        logger.info(f"Deleted {deleted} chunks for source: {source_name}")
        return deleted
    except Exception as e:
        logger.error(f"Failed to delete document {source_name}: {e}")
        return 0

