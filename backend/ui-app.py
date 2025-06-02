import streamlit as st
import tempfile, os, uuid

from ingest.text import extract_text_from_pdf
from ingest.image_ocr import extract_text_from_image
from extract.entity_graph_builder import extract_entities_and_relationships
from graphdb.neo4j_setup import connect_to_neo4j, insert_graph_data, close_connection
from vectordb.qdrant_setup import init_collection, add_document

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MinShould
from graphdb.neo4j_setup import GraphDatabase
from openai import OpenAI
from ingest.utils import chunk_text

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "rag_chunks"
init_collection()

st.title("🧠 Multimodal RAG Assistant (PDF & Image)")


# === File Upload ===
uploaded_file = st.file_uploader(
    "Upload a PDF or Image", type=["pdf", "jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    filetype = uploaded_file.type
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    text = ""
    metadata = {"filename": uploaded_file.name, "filetype": suffix}

    try:
        if "pdf" in filetype:
            result = extract_text_from_pdf(tmp_path)
            text = result["text"]
            metadata.update(result["metadata"])
            st.subheader("📄 PDF Text")
        elif "image" in filetype:
            result = extract_text_from_image(tmp_path)
            text = result["text"]
            metadata.update(result["metadata"])
            st.image(tmp_path, caption="Uploaded Image", use_column_width=True)
            st.subheader("🖼️ Image Text")

        if text:
            st.text_area("📑 Extracted Text", text[:5000], height=300)

            if st.button("Extract Entities & Insert into Neo4j"):
                graph = extract_entities_and_relationships(text)
                st.write("Entities:", graph["entities"])
                st.write("Relationships:", graph["relationships"])

                connect_to_neo4j("bolt://localhost:7687", "neo4j", "strongpassword123")
                insert_graph_data(graph["entities"], graph["relationships"])
                close_connection()
                st.success("✅ Inserted into Neo4j")

            if st.button("Insert Chunks into Qdrant"):
                doc_id = str(uuid.uuid4())
                add_document(doc_id, text, metadata)  # Uses OpenAI embeddings
                st.success("✅ Chunks added to Qdrant")
    finally:
        os.remove(tmp_path)


# === Hybrid Search Logic ===
class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_related_entities(self, source_name, relation_type=None):
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


def get_openai_embedding(text):
    client = OpenAI()
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


def hybrid_search(user_query: str, source_node: str, relation_type: str = None):
    st.write(f"Running hybrid search for: `{user_query}` via `{source_node}`")

    connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "strongpassword123")
    try:
        related = connector.get_related_entities(source_node, relation_type)
        if source_node not in related:
            related.insert(0, source_node)
    except Exception as e:
        st.error(f"Neo4j error: {e}")
        related = []
    finally:
        connector.close()

    qdrant_filter = None
    if related:
        must_conditions = []
        for ent in related:
            must_conditions.append(
                FieldCondition(key="text", match=MatchValue(value="ent"))
            )
        qdrant_filter = Filter(
            should=must_conditions,
            must=None,
            must_not=None,
            min_should=MinShould(conditions=must_conditions, min_count=1),
        )
        print("qdrant_filter")
        print(qdrant_filter)

    try:
        query_vector = get_openai_embedding(user_query)

        results = client.search(
            collection_name="rag_chunks",
            query_vector=query_vector,
            limit=5,
            query_filter=qdrant_filter,
        )

        return results
    except Exception as e:
        st.error(f"Qdrant search error: {e}")
        return []


# === Hybrid Search UI ===
st.markdown("---")
st.subheader("🔍 Ask a Question with Graph Context")

query = st.text_input("Natural language query:")
source_node = st.text_input("Filter using node (optional):", "")
if st.button("Run Hybrid Search"):
    results = hybrid_search(query, source_node.strip())
    st.markdown("### 🔬 Top Matching Chunks")
    for i, r in enumerate(results):
        with st.expander(f"Result {i+1}"):
            st.write(r.payload["text"])
            st.json(r.payload["metadata"])
