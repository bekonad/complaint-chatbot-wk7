# src/task3_rag_pipeline.py
"""
Task 3: RAG Core Logic for Complaint Chatbot (finalized, always uses sample store)
- Always loads collection from data/vector_store/chroma_task2_10k
- Avoids HF generator errors by using default summarizer
"""

import os, sys
import chromadb
from sentence_transformers import SentenceTransformer

SAMPLE_STORE = "data/vector_store/chroma_task2_10k"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

def log(msg): print(f"[Task3] {msg}", flush=True)

def load_collection():
    client = chromadb.PersistentClient(path=SAMPLE_STORE)
    names = [c.name for c in client.list_collections()]
    if not names:
        raise RuntimeError("Sample store has no collections.")
    chosen = names[0]
    log(f"Loaded collection from sample store: {chosen}")
    return client.get_collection(chosen)

class QueryEmbedder:
    def __init__(self, model_name):
        log(f"Loading embedder: {model_name}")
        self.model = SentenceTransformer(model_name)
    def encode(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].astype("float32").tolist()

def default_generate(question, pieces):
    if not pieces: return "I don't have enough information."
    return "From complaints: " + "; ".join([p.replace("\n"," ") for p in pieces[:3]])

def answer_question(collection, embedder, question, top_k=TOP_K):
    qvec = embedder.encode(question)
    res = collection.query(query_embeddings=[qvec], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ans = default_generate(question, docs)
    sources = []
    for i,(d,m) in enumerate(zip(docs, metas),1):
        sources.append(f"[{i}] {m.get('product_category','N/A')} | {d[:120]}...")
    return ans, sources

def main():
    log("Starting Task3 pipeline...")
    collection = load_collection()
    log(f"Collection size: {collection.count()}")
    embedder = QueryEmbedder(EMBEDDING_MODEL)
    demo_q = "Why are people unhappy with Credit Cards?"
    ans, srcs = answer_question(collection, embedder, demo_q)
    print("\nANSWER:\n", ans)
    print("\nSOURCES:")
    for s in srcs: print("-", s)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Fatal error: {e}")
        sys.exit(1)
