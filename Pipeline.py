# run_pipeline.py

"""
Patent Data Processing Pipeline

This script processes patent JSON data through a complete ETL pipeline:
1. Loads raw JSON patent data
2. Normalizes and structures the data
3. Stores data in DuckDB for SQL queries
4. Generates embeddings for semantic search
5. Persists embeddings in ChromaDB vector store

Dependencies:
    - duckdb: SQL analytics database
    - pandas: Data manipulation and analysis
    - sentence_transformers: Text embedding models
    - chromadb: Vector database for embeddings
"""

import json
import hashlib
import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration constants for file paths and database settings
JSON_FILE = "results-2025-07-18T05-11-53.json"  # Source patent data file
DUCKDB_FILE = "patent_data.duckdb"              # Local SQL database file
RAW_TABLE = "raw_patents"                       # Table for unprocessed data
CLEANED_TABLE = "cleaned_patents"               # Table for processed data
VECTOR_COLLECTION = "patent_embeddings"         # ChromaDB collection name
CHROMA_DIR = "./chroma_store"                   # Vector database directory

# STEP 1: Load and parse JSON patent data from file
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

# STEP 2: Extract and normalize patent data from nested JSON structure
records = []
for entry in data["patentdata"]:
    # Extract nested metadata structures
    meta = entry.get("applicationMetaData", {})
    corr = entry.get("correspondenceAddressBag", {})
    
    # Parse inventor information from nested structure
    inventors = meta.get("inventorBag", [])
    inventor_names = [i.get("inventorNameText", "") for i in inventors]
    
    # Create normalized record with flattened structure
    record = {
        "application_number": entry.get("applicationNumberText"),
        "filing_date": meta.get("filingDate"),
        "entity_type": meta.get("entityStatusData", {}).get("businessEntityStatusCategory"),
        "first_inventor_flag": meta.get("firstInventorToFileIndicator"),
        "inventors": ", ".join(inventor_names),
        "correspondence_text": json.dumps(corr),  # Serialize nested object
    }
    
    # Generate summary text for embedding and search
    summary_text = f"{record['inventors']} | {record['entity_type']} | {record['filing_date']}"
    record["summary_text"] = summary_text
    
    # Create unique fingerprint for deduplication using SHA256 hash
    fingerprint_input = f"{record['application_number']}-{summary_text}"
    record["source_fingerprint"] = hashlib.sha256(fingerprint_input.encode()).hexdigest()
    
    # Calculate basic quality score based on data completeness
    record["quality_score"] = len(inventor_names) + int(bool(record["filing_date"]))
    
    records.append(record)

# STEP 3: Persist normalized data to DuckDB for SQL analytics
df = pd.DataFrame(records)
con = duckdb.connect(DUCKDB_FILE)

# Drop existing tables to ensure clean state
con.execute(f"DROP TABLE IF EXISTS {RAW_TABLE}")
con.execute(f"DROP TABLE IF EXISTS {CLEANED_TABLE}")

# Create raw data table from DataFrame
con.execute(f"CREATE TABLE {RAW_TABLE} AS SELECT * FROM df")

# STEP 4: Clean and standardize text fields for consistent processing
df_clean = df.copy()
# Normalize text fields: lowercase and strip whitespace
df_clean["summary_text"] = df_clean["summary_text"].str.lower().str.strip()
df_clean["inventors"] = df_clean["inventors"].str.lower().str.strip()

# Create cleaned data table
con.execute(f"CREATE TABLE {CLEANED_TABLE} AS SELECT * FROM df_clean")

# STEP 5: Generate embeddings and store in persistent vector database
# Initialize sentence transformer model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Clean slate: delete existing collection if it exists
if VECTOR_COLLECTION in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(VECTOR_COLLECTION)

# Create new collection for patent embeddings
collection = chroma_client.create_collection(name=VECTOR_COLLECTION)

# Prepare data for embedding: texts, metadata, and embeddings
texts = df_clean["summary_text"].tolist()
metadatas = df_clean[["application_number", "filing_date", "entity_type", "source_fingerprint", "quality_score"]].to_dict(orient="records")

# Generate embeddings with progress tracking
embeddings = model.encode(texts, show_progress_bar=True)

# Add all data to ChromaDB collection
collection.add(
    documents=texts,         # Original text for reference
    embeddings=embeddings,   # Vector representations
    metadatas=metadatas,    # Searchable metadata
    ids=[str(i) for i in range(len(texts))]  # Unique identifiers
)

# Data automatically persists to disk with PersistentClient

print("✅ Pipeline run complete. Vector DB and DuckDB are ready.")