# embeddings.py
import os
import json
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/chunks")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

print(f"[embeddings] loading model {EMBED_MODEL_NAME} ...")
_model = SentenceTransformer(EMBED_MODEL_NAME)

def load_chunks() -> List[dict]:
    chunks = []
    if not os.path.exists(CHUNKS_DIR):
        print("[embeddings] no chunks dir:", CHUNKS_DIR)
        return chunks
    for fn in sorted(os.listdir(CHUNKS_DIR)):
        if fn.endswith(".json"):
            path = os.path.join(CHUNKS_DIR, fn)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                # expected fields: id, document, text
                chunks.append(obj)
    print(f"[embeddings] loaded {len(chunks)} chunk files")
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    """Return float32 embeddings (N, D). Normalized for cosine (IP) search."""
    if not texts:
        return np.zeros((0, _model.get_sentence_embedding_dimension()), dtype="float32")
    emb = _model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # normalize to unit vectors for inner-product == cosine similarity
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    emb = emb / norm
    return emb.astype("float32")
