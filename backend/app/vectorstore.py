# vectorstore.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import json
import faiss
import numpy as np
from backend.app.embeddings import load_chunks, embed_texts

INDEX_DIR = os.getenv("INDEX_DIR", "data/indexes")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")

def build_faiss_index(use_hnsw: bool = False):
    os.makedirs(INDEX_DIR, exist_ok=True)
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    if len(texts) == 0:
        print("[vectorstore] no chunks to index")
        return

    embs = embed_texts(texts)
    dim = embs.shape[1]
    print(f"[vectorstore] embedding dim {dim}, building index (n={len(embs)})")

    if use_hnsw:
        # HNSW for large collections (requires extra tuning)
        index = faiss.IndexHNSWFlat(dim, 32)  # M=32
        index.hnsw.efConstruction = 200
        index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        # simple inner-product flat index — fine for medium-sized corpora
        index = faiss.IndexFlatIP(dim)

    index.add(embs)
    faiss.write_index(index, INDEX_PATH)
    print("[vectorstore] saved index to", INDEX_PATH)

    metadata = {"chunks": chunks}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("[vectorstore] saved metadata to", META_PATH)

def load_faiss_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Index or metadata not found. Run build_faiss_index() first.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta
