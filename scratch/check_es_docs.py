# scratch/check_es_docs.py
import os
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX_NAME = "socratic_docs"

def main():
    client = Elasticsearch(ES_URL)
    if not client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' does not exist.")
        return
        
    # Search all documents (max 1000)
    resp = client.search(
        index=INDEX_NAME,
        body={
            "query": {"match_all": {}},
            "size": 1000,
            "_source": ["source", "page", "text"]
        }
    )
    
    hits = resp["hits"]["hits"]
    print(f"Total chunks in Elasticsearch: {len(hits)}")
    
    sources = {}
    for h in hits:
        source = h["_source"].get("source", "Unknown Source")
        sources[source] = sources.get(source, 0) + 1
        
    print("\n--- Document Chunk Counts by Source ---")
    for src, count in sources.items():
        print(f" * {src}: {count} chunks")
        
if __name__ == "__main__":
    main()
