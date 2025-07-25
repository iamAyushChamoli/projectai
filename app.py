# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

# Directory where ChromaDB stores persistent vector database files
# Ensures embeddings survive application restarts
CHROMA_DIR = "./chroma_store"

# Load sentence transformer model for text-to-embedding conversion
# all-MiniLM-L6-v2 produces 384-dim vectors, good balance of speed and accuracy
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize persistent ChromaDB client to store embeddings on disk
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Get or create collection to store patent document embeddings and metadata
collection = chroma_client.get_or_create_collection("patent_embeddings")

# Initialize FastAPI application
app = FastAPI()

class QueryInput(BaseModel):
    # Natural language search query to be converted to embeddings
    query: str

@app.post("/search")
def search_patents(request: QueryInput):
    # Convert user query to embedding vector using same model as indexed documents
    query_embedding = model.encode(request.query)
    
    # Find top 3 most similar patent documents using cosine similarity
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    # Build response by combining document text with metadata for each result
    response = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        response.append({
            "summary": doc,
            "application_number": meta.get("application_number"),
            "filing_date": meta.get("filing_date"),
            "entity_type": meta.get("entity_type"),
            "quality_score": meta.get("quality_score"),
            "source_fingerprint": meta.get("source_fingerprint")
        })
    return {"results": response}

# Start development server with:
# uvicorn app:app --host 0.0.0.0 --port 8000