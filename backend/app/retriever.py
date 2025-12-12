# retriever.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
from backend.app.vectorstore import load_faiss_index
from backend.app.embeddings import embed_texts
from backend.app.reranker import rerank_chunks
from backend.app.config import settings

# load index at import-time
_index = None
_metadata = None


def ensure_index():
    global _index, _metadata
    if _index is None:
        _index, _metadata = load_faiss_index()
    return _index, _metadata


def embed_query(query: str):
    emb = embed_texts([query])
    return emb.astype("float32")


def retrieve_top_k(query: str, k: int = None):
    """
    Retrieve top-k chunks (after reranking).
    Returns list of:
    { chunk_idx, chunk_id, document, text, score, rerank_score }
    """
    if k is None:
        k = settings.TOP_K

    index, meta = ensure_index()
    chunks = meta["chunks"]

    # --- embed query ---
    qv = embed_query(query)

    # --- wider FAISS search before reranking ---
    search_k = max(k * 3, k + 5)
    D, I = index.search(qv, search_k)

    D = D[0]
    I = I[0]

    # Build FAISS candidates
    candidates = []
    for dist, idx in zip(D, I):
        if idx < 0 or idx >= len(chunks):
            continue

        c = chunks[idx]

        candidates.append({
            "chunk_idx": idx,
            "chunk_id": c.get("id"),
            "document": c.get("document"),
            "text": c.get("text"),
            "score": float(dist)
        })

    # --- reranking using cross-encoder ---
    reranked = rerank_chunks(query, candidates)

    # select only top-k
    return reranked[:k]
