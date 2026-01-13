# src/build_full_chroma_from_parquet.py
import os
import pyarrow.parquet as pq
import chromadb
import pandas as pd

PARQUET_FILE = "data/complaint_embeddings.parquet"
SAVE_FOLDER = "vector_store/full_chroma"
COLLECTION_NAME = "full_complaints"

def log(msg): print(f"[Build] {msg}")

def main():
    if not os.path.exists(PARQUET_FILE):
        raise FileNotFoundError(f"Missing {PARQUET_FILE}")
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    client = chromadb.PersistentClient(path=SAVE_FOLDER)

    # Create or get collection
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in names:
        collection = client.get_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        log(f"Using existing collection: {COLLECTION_NAME}")
    else:
        collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        log(f"Created collection: {COLLECTION_NAME}")

    dataset = pq.ParquetFile(PARQUET_FILE)
    log(f"Row groups: {dataset.num_row_groups}")

    for rg in range(dataset.num_row_groups):
        table = dataset.read_row_group(rg)
        df = table.to_pandas()
        log(f"Row group {rg}: {len(df)} rows")

        # Detect columns
        text_col = "chunk" if "chunk" in df.columns else ("text" if "text" in df.columns else None)
        if text_col is None or "embedding" not in df.columns:
            raise ValueError(f"Expected columns 'chunk' or 'text', and 'embedding'. Found: {df.columns.tolist()}")

        docs = df[text_col].astype(str).tolist()
        embs = [list(map(float, e)) for e in df["embedding"]]
        metas = df.drop([text_col, "embedding"], axis=1).to_dict("records")

        # Unique ids per row group
        ids = [f"{rg}_{i}" for i in range(len(df))]

        collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
        log(f"Added {len(df)} docs from row group {rg}")

    log(f"Build complete. Collection size: {collection.count()}")

if __name__ == "__main__":
    main()
