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

from ingest.utils import chunk_text
from uuid import uuid4


def add_document(id: str, text: str, metadata: dict):
    chunks = chunk_text(text)

    points = []
    document_uuid = str(uuid4())  # Generate a UUID for the document
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        chunk_meta = metadata.copy()
        chunk_meta["chunk_index"] = i
        points.append(
            PointStruct(
                id=i,  # Use an integer for the chunk ID
                vector=embedding,
                payload={"text": chunk, "metadata": chunk_meta},
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query: str, top_k=5, filter_terms=None):
    query_vector = model.encode(query).tolist()

    search_filter = None
    if filter_terms:
        search_filter = {
            "must": [
                {
                    "key": "metadata.filename",
                    "match": {"any": filter_terms}
                }
            ]
        }

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter
    )
    return results
