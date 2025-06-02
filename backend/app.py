import sys

print(">>> Python Path:", sys.executable)

from ingest.text import extract_text_from_pdf
from ingest.image_ocr import extract_text_from_image
from ingest.audio_transcribe import transcribe_audio
from extract.entity_graph_builder import extract_entities_and_relationships


if __name__ == "__main__":
    # --- PDF Test ---
    pdf_data = extract_text_from_pdf("data/sample_files/sample.pdf")
    # print("\nPDF METADATA:", pdf_data["metadata"])
    # print("PDF TEXT:", pdf_data["text"][:500])

    # --- Image Test ---
    image_data = extract_text_from_image("data/sample_files/sample_image.jpg")
    print("\nIMAGE METADATA:", image_data["metadata"])
    print("IMAGE TEXT:", image_data["text"][:500])

    # --- Audio Test ---
    audio_data = transcribe_audio("data/sample_files/sample_audio.mp3")
    print("\nAUDIO METADATA:", audio_data["metadata"])
    print("AUDIO TEXT:", audio_data["text"][:500])

    # --- Entity Extraction Test (choose one source) ---
    sample_text = pdf_data["text"]  # Or use image_data["text"], audio_data["text"]
    graph = extract_entities_and_relationships(sample_text)

    print("\nENTITIES:", graph["entities"])
    print("RELATIONSHIPS:")
    for rel in graph["relationships"]:
        print("-", rel)


from graphdb.neo4j_setup import connect_to_neo4j, insert_graph_data, close_connection

# Connect to Neo4j
connect_to_neo4j("bolt://localhost:7687", "neo4j", "strongpassword123")

# Insert to Neo4j
insert_graph_data(graph["entities"], graph["relationships"])

# Close connection
close_connection()


from vectordb.qdrant_setup import init_collection, add_document, search
import uuid

# --- Initialize vector store
init_collection()

# --- Choose a text block (e.g., from PDF)
doc_text = pdf_data["text"]
doc_meta = pdf_data["metadata"]

# --- Add to Qdrant
doc_id = str(uuid.uuid4())  # Generate a valid UUID for the document ID
add_document(id=doc_id, text=doc_text, metadata=doc_meta)


from graphdb.neo4j_setup import get_related_entities
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687", auth=("neo4j", "strongpassword123")
)


from graphdb.neo4j_setup import connect_to_neo4j, get_related_entities, close_connection
from vectordb.qdrant_setup import search
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MinShould
from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer


# -----------------------
# 1) A small Neo4j helper.
# -----------------------
class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_related_entities(self, source_name, relation_type=None):
        """
        Return a list of node-names connected from `source_name`.
        If relation_type is None, return all outgoing neighbors.
        """
        with self.driver.session() as session:
            if relation_type:
                cypher = f"""
                MATCH (a {{name: $src}})-[r:`{relation_type}`]->(b)
                RETURN DISTINCT b.name AS name
                """
                result = session.run(cypher, src=source_name)
            else:
                cypher = """
                MATCH (a {name: $src})-[r]->(b)
                RETURN DISTINCT b.name AS name
                """
                result = session.run(cypher, src=source_name)

            return [record["name"] for record in result]


# -----------------------
# 2) The hybrid_search function
# -----------------------
def hybrid_search(user_query: str, source_node: str, relation_type: str = None):
    """
    1) Look up neighbors of `source_node` in Neo4j.
    2) Force-include `source_node` itself.
    3) Wrap each in wildcard '*…*' and build a Filter.
    4) Compute query_vector from user_query.
    5) Call Qdrant search with that filter.
    6) Print Neo4j list + Qdrant hits.
    """
    print(f"\n--- Hybrid Search for '{user_query}' via '{source_node}' ---")

    # 1) Neo4j lookup
    connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "strongpassword123")
    try:
        related = connector.get_related_entities(source_node, None)
        print("Entities from Neo4j:", related)
        # Always include the source itself so "*source_node*" will match at least one chunk
        if source_node not in related:
            related.insert(0, source_node)
        print("Entities from Neo4j:", related)
    except Exception as e:
        print("❌ Neo4j error:", e)
        related = []
    finally:
        connector.close()

    # 2) Build a Qdrant Filter with one FieldCondition per entity, each wildcarded
    if related:
        must_conditions = []
        for ent in related:
            wildcard = f"*{ent}*"
            must_conditions.append(
                FieldCondition(key="text", match=MatchValue(value=wildcard))
            )
        qdrant_filter = Filter(
            should=must_conditions,
            must=None,
            must_not=None,
            min_should=MinShould(conditions=must_conditions, min_count=1),
        )
    else:
        # If no Neo4j entities or error, do a pure semantic search (no filter)
        qdrant_filter = None
    # print("drant_filter")
    # print(qdrant_filter)

    # 3) Compute query_vector
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(user_query).tolist()

    # 4) Qdrant client + search
    client = QdrantClient(host="localhost", port=6333)
    try:
        results = client.search(
            collection_name="rag_chunks",
            query_vector=query_vector,
            limit=5,
            query_filter=None,  # pass the Filter object here
        )
        print("\nQDRANT RESULTS:")
        for hit in results:
            text_payload = hit.payload.get("text", "<no text>")
            print(f"– Score {hit.score:.3f} | {text_payload[:120]}…")
    except Exception as e:
        print("❌ Qdrant search error:", e)


# -----------------------
# 3) Usage examples
# -----------------------

# Example A: filter on “Hybrid Search”
# hybrid_search(user_query="What do cats like", source_node="cow")

# Example B: filter on “knowledge graph”
hybrid_search(
    user_query="What is a searchable knowledge graph?", source_node="knowledge graph"
)

# Example C: filter on the full long title (copy exactly from your chunk)
long_title = "Multimodal Enterprise RAG – Leveraging Knowledge Graphs and Hybrid Search"
hybrid_search(user_query="Describe this RAG challenge", source_node=long_title)


# Example usage
# hybrid_search(
#     user_query="What is this 72-Hour Technical Challenge about?",
#     structured_filter_source="Multimodal Enterprise RAG – Leveraging Knowledge Graphs and Hybrid Search",
#     relation_type=None,
# )
# hybrid_search(
#     user_query="Describe the RAG system workflow",
#     source_node="Enterprise Retrieval-Augmented Generation (RAG) system",
#     relation_type=None,
# )
# hybrid_search(
#     user_query="What is this 72-Hour Technical Challenge about?",
#     structured_filter_source="Multimodal Enterprise RAG – Leveraging Knowledge Graphs and Hybrid Search",
#     relation_type=None,
# )
