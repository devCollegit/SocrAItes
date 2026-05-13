
import chromadb
from chromadb.utils import embedding_functions
import os

print("Starting chromadb test...")
try:
    embeddings = embedding_functions.DefaultEmbeddingFunction()
    print("Embedding function created.")
    client = chromadb.PersistentClient(path="./test_chroma")
    print("Client created.")
    collection = client.get_or_create_collection(name="test", embedding_function=embeddings)
    print("Collection created.")
    collection.add(ids=["1"], documents=["hello world"])
    print("Document added.")
    results = collection.query(query_texts=["hello"], n_results=1)
    print("Query results:", results)
except Exception as e:
    print("Error:", e)
