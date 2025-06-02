Setup Instructions
1. Clone the repository
bash
CopyEdit
git clone https://github.com/oopilyk/aparavitask.git
cd aparavitask

2. Create & activate virtual environment
bash
CopyEdit
python3 -m venv rag-env
source rag-env/bin/activate

3. Install system dependencies (macOS)
bash
CopyEdit
brew install tesseract ffmpeg

4. Install Python packages
bash
CopyEdit
pip install --upgrade pip
pip install streamlit PyPDF2 openai qdrant-client python-dotenv pytesseract pillow

For audio support:
bash
CopyEdit
pip install torch
pip install git+https://github.com/openai/whisper.git

For image OCR enhancement:
bash
CopyEdit
pip install easyocr

5. Set OpenAI API key
Create a .env file at the root of your repo:
env
CopyEdit
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx


🧩 Neo4j Setup
Option A: Use Neo4j Desktop
Download Neo4j Desktop and create a local DB with:
Bolt URL: bolt://localhost:7687


Username: neo4j


Password: strongpassword123


Useful Cypher commands:
cypher
CopyEdit
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25;
MATCH (n) DETACH DELETE n;  -- wipes all data


📦 Qdrant Setup
Start local Qdrant:
bash
CopyEdit
docker run -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

Web UI: http://localhost:6333


REST API: http://localhost:6333


gRPC: localhost:6334



🚀 Run the App

1. Backend Tests (CLI)
bash
CopyEdit
python backend/app.py

This will:
Extract text from PDF/image/audio under data/sample_files


Insert graph to Neo4j


Chunk and embed into Qdrant


Run sample hybrid queries using your graph + vector DB


2. Frontend UI (Streamlit)
bash
CopyEdit
streamlit run backend/ui-app.py

Upload a PDF or image.


Use buttons: Extract text + insert to Neo4j + Qdrant.


Ask a question via hybrid search powered by OpenAI embeddings.


