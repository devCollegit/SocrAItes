# src/rag/vectorstore.py
"""Vector store wrapper for SocrAItes.

Uses ChromaDB (local) to store document chunks and perform similarity
retrieval. This is a minimal implementation suitable for the MVP. In a
production setting you would configure persistence directory, embedding
model, and optionally a reranker.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Environment variable name for the persistence folder (can be overridden)
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

if os.getenv("OPENAI_API_KEY"):
    embeddings = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
else:
    # Use default embedding function if API key is missing
    embeddings = embedding_functions.DefaultEmbeddingFunction()


def get_client() -> PersistentClient:
    """Return a Chroma PersistentClient, creating the directory if needed."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    return PersistentClient(path=PERSIST_DIR)


def get_collection(name: str = "socratic_docs"):
    """Retrieve (or create) a collection for storing document chunks.

    The collection uses the ``embeddings`` function defined above.
    Uses get_or_create_collection for compatibility with ChromaDB v0.5+.
    """
    client = get_client()
    return client.get_or_create_collection(name=name, embedding_function=embeddings)


def add_documents(docs: List[str], metadatas: List[dict] | None = None, ids: List[str] | None = None) -> None:
    """Add a list of document *chunks* to the vector store.

    Each chunk becomes a separate record. ``metadatas`` can contain source
    information like ``{"source": "Week5.pdf", "page": 12}``.
    ``ids`` can be provided explicitly to avoid duplicates; if omitted,
    auto-generated IDs are used.
    """
    collection = get_collection()
    existing_ids = set(collection.get()["ids"])
    
    if ids is None:
        import uuid
        ids = [str(uuid.uuid4()) for _ in docs]
    
    # Filter out already-existing IDs to prevent duplicate insertion
    new_docs, new_metas, new_ids = [], [], []
    for doc, meta, id_ in zip(docs, metadatas or [{}] * len(docs), ids):
        if id_ not in existing_ids:
            new_docs.append(doc)
            new_metas.append(meta)
            new_ids.append(id_)
    
    if new_docs:
        collection.add(ids=new_ids, documents=new_docs, metadatas=new_metas)
    
    return len(new_docs)


def query(query_text: str, k: int = 5) -> List[Tuple[str, float]]:
    """Return the top‑k most similar document chunks for *query_text*.

    Returns a list of ``(document, distance)`` tuples.
    """
    collection = get_collection()
    results = collection.query(query_texts=[query_text], n_results=k)
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return list(zip(docs, dists))
