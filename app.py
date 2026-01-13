# app.py
import gradio as gr, os
import chromadb
from sentence_transformers import SentenceTransformer

FULL_STORE = "vector_store/full_chroma"
SAMPLE_STORE = "data/vector_store/chroma_task2_10k"

def get_collection():
    if os.path.exists(FULL_STORE):
        c = chromadb.PersistentClient(path=FULL_STORE).list_collections()
        if c: return c[0]
    if os.path.exists(SAMPLE_STORE):
        c = chromadb.PersistentClient(path=SAMPLE_STORE).list_collections()
        if c: return c[0]
    raise RuntimeError("No store found.")

collection = get_collection()
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def ask(q):
    v = embedder.encode([q], normalize_embeddings=True)[0].astype("float32").tolist()
    res = collection.query(query_embeddings=[v], n_results=5)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    answer = "From complaints: " + "; ".join([d.replace("\n"," ") for d in docs[:3]]) if docs else "I don't have enough information."
    sources = [f"[{i+1}] {m.get('product_category','N/A')} | {docs[i][:140]}..." for i,m in enumerate(metas[:3])]
    return answer, "\n".join(sources)

ui = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, label="Ask a question"),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
    title="CrediTrust Complaint RAG"
)

if __name__ == "__main__":
    ui.launch()
