## Project Overview

The pipeline demonstrates the following capabilities:

- Structured data ingestion from a raw JSON dataset of patent applications
- Data normalization and field flattening
- Source fingerprinting using SHA256 hashing for traceability
- Quality scoring heuristics based on inventor count and metadata completeness
- Storage of raw and cleaned data in a DuckDB database
- Embedding of key descriptive fields using a sentence-transformer model
- Persistence of semantic vectors in a ChromaDB vector database
- Deployment of a search endpoint using FastAPI to retrieve semantically relevant patents

---

## Repository Structure

```
├── run_pipeline.py          # Core pipeline: parse, clean, embed, persist
├── app.py                   # FastAPI application exposing /search endpoint
├── results-*.json           # Source JSON dataset
├── patent_data.duckdb       # DuckDB database file
├── chroma_store/            # ChromaDB persistent store
├── demo_results.json        # Example search query and output
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## Setup Instructions

### 1. Environment Setup

```bash
python -m venv env
source env/bin/activate      # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Execute the Pipeline

```bash
python run_pipeline.py
```

This step will:

- Ingest and normalize the dataset
- Create `raw_patents` and `cleaned_patents` tables in DuckDB
- Generate semantic embeddings from descriptive metadata
- Store vectorized data and metadata in a persistent ChromaDB store

### 3. Launch the Search API

```bash
uvicorn app:app --reload
```

This launches a local FastAPI server exposing a `/search` endpoint.

### 4. Example Query

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "artificial intelligence patents"}'
```

---

## Sample Output

```json
{
  "results": [
    {
      "summary": "wilfred gomes, abhishek a. sharma, ...",
      "application_number": "19066538",
      "filing_date": "2025-02-28",
      "entity_type": "Regular Undiscounted",
      "quality_score": 5,
      "source_fingerprint": "e84c17..."
    },
    ...
  ]
}
```

---

## Traceability and Design Principles

- **Fingerprinting**: Every record includes a deterministic SHA256 fingerprint based on core metadata, enabling deduplication and full source traceability.
- **Scoring**: A basic quality score is assigned using heuristics such as inventor count and date presence.
- **Embeddings**: Sentence-transformer model (`all-MiniLM-L6-v2`) is used to convert descriptive text into embeddings.
- **Search**: The FastAPI service retrieves top-K results using cosine similarity from ChromaDB.

---

## Final Notes

This implementation demonstrates readiness for building robust, traceable, and scalable ingestion pipelines integrated with semantic search capabilities. All components use production-capable, open-source technologies.
