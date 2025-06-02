from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from ingest.utils import chunk_text
from uuid import uuid4
import openai
import os

# Ensure the OpenAI API key is loaded
openai.api_key = os.getenv("OPENAI_API_KEY")

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "rag_chunks"
VECTOR_DIM = 1536  # OpenAI text-embedding-3-small output dimension


def get_openai_embedding(text):
    """
    Get the OpenAI embedding for a given text using text-embedding-3-small model.
    """
    response = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def init_collection():
    """
    Create the Qdrant collection with the correct vector dimension and distance,
    if it doesn't already exist.
    """
    if COLLECTION_NAME not in client.get_collections().collections:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def add_document(id: str, text: str, metadata: dict):
    """
    Chunk the text, embed each chunk with OpenAI, and upsert into Qdrant.
    """
    chunks = chunk_text(text)

    points = []
    for i, chunk in enumerate(chunks):
        embedding = get_openai_embedding(chunk)
        chunk_meta = metadata.copy()
        chunk_meta["chunk_index"] = i
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={"text": chunk, "metadata": chunk_meta},
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query: str, top_k=5, filter_terms=None):
    """
    Perform a semantic search in Qdrant using OpenAI embeddings and optional filtering.
    """
    query_vector = get_openai_embedding(query)

    search_filter = None
    if filter_terms:
        search_filter = {
            "must": [{"key": "metadata.filename", "match": {"any": filter_terms}}]
        }

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter,
    )
    return results
