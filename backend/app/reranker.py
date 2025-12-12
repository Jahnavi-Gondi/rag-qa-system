# backend/app/reranker.py

from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_chunks(query, retrieved_chunks):
    """
    Reranks retrieved chunks by semantic relevance.
    Ensures ALL text passed to reranker is clean string.
    """

    clean_items = []
    for c in retrieved_chunks:
        text = str(c.get("text", ""))    # force string
        clean_items.append({
            "chunk": c,
            "text": text
        })

    pairs = [(query, item["text"]) for item in clean_items]

    scores = _reranker.predict(pairs, show_progress_bar=False)

    # Attach scores back
    for i, score in enumerate(scores):
        clean_items[i]["chunk"]["rerank_score"] = float(score)

    # Sort desc by rerank score
    sorted_items = sorted(clean_items, key=lambda x: x["chunk"]["rerank_score"], reverse=True)

    # Return only original chunk metadata
    return [item["chunk"] for item in sorted_items]
