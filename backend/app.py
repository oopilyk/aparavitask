from ingest.text import extract_text_from_pdf
from ingest.image_ocr import extract_text_from_image
from ingest.audio_transcribe import transcribe_audio
from extract.entity_graph_builder import extract_entities_and_relationships

if __name__ == "__main__":
    # --- PDF Test ---
    pdf_data = extract_text_from_pdf("data/sample_files/sample.pdf")
    print("\nPDF METADATA:", pdf_data["metadata"])
    print("PDF TEXT:", pdf_data["text"][:500])

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

sample_graph = {
    "entities": ["Apple", "Beats"],
    "relationships": [
        {"source": "Apple", "relation": "acquired", "target": "Beats", "extra": {"year": "2014", "amount": "$3B"}}
    ]
}

# Connect to Neo4j
connect_to_neo4j("bolt://localhost:7687", "neo4j", "strongpassword123")

# Insert to Neo4j
insert_graph_data(sample_graph["entities"], sample_graph["relationships"])

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

# --- Run a similarity search
query = "What company did Apple acquire?"
results = search(query)

print("\nQDRANT SEARCH RESULTS:")
for r in results:
    print("-", r.payload["text"][:100], "...\n")
