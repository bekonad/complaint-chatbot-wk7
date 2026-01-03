# src/task2_chunk_embed_index.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ─── SETTINGS ────────────────────────────────────────────────────────────────
SAMPLE_SIZE       = 10000                  # start with 10k – safe for most machines
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "data/vector_store/chroma_task2_10k"   # inside data/ as you wanted
COLLECTION_NAME   = "complaints_10k_sample"

# Make sure the folder exists
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# ─── 1. Load filtered data ──────────────────────────────────────────────────
print("Loading filtered complaints from Task 1...")
csv_path = "data/processed/filtered_complaints.csv"   # ← your actual path
df = pd.read_csv(csv_path)

# Quick safety check
expected_cols = ["Product", "Consumer complaint narrative", "Complaint ID"]
missing = [col for col in expected_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Check your Task 1 output.")

print("\nColumns found:", df.columns.tolist())
print("\nProduct distribution in full filtered data:")
print(df["Product"].value_counts(normalize=True).round(4))

# ─── 2. Stratified sample ───────────────────────────────────────────────────
print("\nCreating stratified sample of", SAMPLE_SIZE, "complaints...")
df_sample, _ = train_test_split(
    df,
    train_size=SAMPLE_SIZE,
    stratify=df["Product"],
    random_state=42
)

print("Product distribution in sample:")
print(df_sample["Product"].value_counts(normalize=True).round(4))

# ─── 3. Chunking ─────────────────────────────────────────────────────────────
print("\nChunking complaint narratives...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = []
metadatas = []
ids = []

for idx, row in df_sample.iterrows():
    # Use exact column name from your CSV
    narrative = row.get("Consumer complaint narrative", "")
    if not isinstance(narrative, str) or len(narrative.strip()) == 0:
        continue

    splits = text_splitter.split_text(narrative)
    total_chunks_this_complaint = len(splits)

    for chunk_idx, chunk_text in enumerate(splits):
        # Use exact column name for ID
        complaint_id = str(row.get("Complaint ID", f"unknown_{idx}"))
        chunk_id = f"{complaint_id}_c{chunk_idx:03d}"

        ids.append(chunk_id)
        chunks.append(chunk_text)

        metadatas.append({
            "complaint_id":     complaint_id,
            "product_category": row["Product"],               # key matches pre-built spec
            "product":          row.get("Sub-product", ""),   # optional but useful
            "issue":            row.get("Issue", ""),
            "sub_issue":        row.get("Sub-issue", ""),
            "company":          row.get("Company", ""),
            "state":            row.get("State", ""),
            "date_received":    str(row.get("Date received", "")),
            "chunk_index":      chunk_idx,
            "total_chunks":     total_chunks_this_complaint
        })

print(f"→ Created {len(chunks):,} chunks from {len(df_sample)} complaints")

# ─── 4. Embeddings ───────────────────────────────────────────────────────────
print("Generating embeddings (this may take 5–25 minutes)...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
embeddings = embedder.encode(
    chunks,
    show_progress_bar=True,
    batch_size=64   # smaller batch = less memory usage
)

# ─── 5. Save to ChromaDB ─────────────────────────────────────────────────────
print("Creating and persisting Chroma collection...")
client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

# Clean up old collection if re-running
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Add in batches to respect ChromaDB max batch size (~5k)
BATCH_SIZE = 5000
total_chunks = len(chunks)
print(f"Adding {total_chunks:,} chunks in batches of {BATCH_SIZE}...")

for i in range(0, total_chunks, BATCH_SIZE):
    batch_start = i
    batch_end = min(i + BATCH_SIZE, total_chunks)
    
    print(f"  Adding batch {batch_start:,} – {batch_end:,} ({batch_end - batch_start:,} items)")
    
    collection.add(
        ids=ids[batch_start:batch_end],
        embeddings=embeddings[batch_start:batch_end].tolist(),
        metadatas=metadatas[batch_start:batch_end],
        documents=chunks[batch_start:batch_end]
    )

print(f"\nSuccess!")
print(f"Vector store saved → {VECTOR_STORE_PATH}")
print(f"Collection name: {COLLECTION_NAME}")
print(f"Total chunks stored: {collection.count()}")