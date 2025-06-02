import streamlit as st
from ingest.text import extract_text_from_pdf
from ingest.image_ocr import extract_text_from_image
from extract.entity_graph_builder import extract_entities_and_relationships
from graphdb.neo4j_setup import connect_to_neo4j, insert_graph_data, close_connection

# from vectordb.qdrant_setup import init_collection, add_document, search
import tempfile
import os
import uuid

st.title("🧠 Multimodal RAG Assistant (PDF & Image)")

init_collection()

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
    metadata = {
        "filename": uploaded_file.name,
        "filetype": suffix,
    }

    try:
        if "pdf" in filetype:
            text = extract_text_from_pdf(tmp_path)
            st.subheader("📄 PDF Text")
        elif "image" in filetype:
            text = extract_text_from_image(tmp_path)
            st.image(tmp_path, caption="Uploaded Image", use_column_width=True)
            st.subheader("🖼️ Image Text")

        if text:
            st.text_area("📑 Extracted Text", text[:1500], height=300)

            # --- Entity Extraction ---
            if st.button("Extract Entities & Insert into Neo4j"):
                graph = extract_entities_and_relationships(text)
                st.write("Entities:", graph["entities"])
                st.write("Relationships:", graph["relationships"])

                # Insert into Neo4j
                connector = connect_to_neo4j(
                    "bolt://localhost:7687", "neo4j", "strongpassword123"
                )
                connector.insert_graph_data(graph["entities"], graph["relationships"])
                connector.close()
                st.success("✅ Inserted into Neo4j")

            # # --- Add to Qdrant ---
            # if st.button("Insert Chunks into Qdrant"):
            #     doc_id = str(uuid.uuid4())
            #     add_document(doc_id, text, metadata)
            #     st.success("✅ Chunks added to Qdrant")

    finally:
        os.remove(tmp_path)

# --- Hybrid Search UI ---
st.markdown("---")
st.subheader("🔍 Ask a Question")

query = st.text_input("Natural language query:")
if st.button("Run Hybrid Search"):
    results = search(query)
    st.markdown("### 🔬 Top Matching Chunks")
    for i, r in enumerate(results):
        with st.expander(f"Result {i+1}"):
            st.write(r.payload["text"])
            st.json(r.payload["metadata"])
