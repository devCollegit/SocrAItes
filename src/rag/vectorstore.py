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

    success, _ = bulk(client, actions, raise_on_error=False)
    return success


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """여러 검색 결과 리스트를 RRF로 합산합니다. score = Σ 1/(k + rank)"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def query(query_text: str, k: int = 5) -> List[Tuple[str, float]]:
    ensure_index()
    client = get_client()

    query_vector = embed_query(query_text)
    window = k * 4

    # BM25 검색
    bm25_resp = client.search(
        index=INDEX_NAME,
        body={"query": {"match": {"text": {"query": query_text}}}, "size": window},
    )
    bm25_hits = {h["_id"]: h["_source"]["text"] for h in bm25_resp["hits"]["hits"]}
    bm25_ranking = list(bm25_hits.keys())

    # Dense KNN 검색
    knn_resp = client.search(
        index=INDEX_NAME,
        body={"knn": {"field": "dense_vector", "query_vector": query_vector, "k": window, "num_candidates": window * 2}, "size": window},
    )
    knn_hits = {h["_id"]: h["_source"]["text"] for h in knn_resp["hits"]["hits"]}
    knn_ranking = list(knn_hits.keys())

    # RRF 합산
    all_docs = {**bm25_hits, **knn_hits}
    rrf_scores = _rrf([bm25_ranking, knn_ranking])
    top_k = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    return [(all_docs[doc_id], score) for doc_id, score in top_k if doc_id in all_docs]
