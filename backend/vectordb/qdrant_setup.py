from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast

COLLECTION_NAME = "rag_chunks"

def init_collection():
    if COLLECTION_NAME not in client.get_collections().collections:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def add_document(id: str, text: str, metadata: dict):
    embedding = model.encode(text).tolist()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=id,
                vector=embedding,
                payload={"text": text, "metadata": metadata}
            )
        ]
    )

def search(query: str, top_k=5):
    query_vector = model.encode(query).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return results
