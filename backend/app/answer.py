# backend/app/answer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
from backend.app.retriever import retrieve_top_k
from backend.app.config import settings
from backend.app.llm import call_llm_sync
from backend.app.reranker import rerank_chunks


def build_context(chunks):
    """
    Build clean context from reranked chunks.
    """
    parts = []
    for c in chunks:
        parts.append(f"[{c['document']}]\n{c['text']}")
    return "\n\n".join(parts)


def answer_query(query: str, top_k: int = settings.TOP_K):
    """
    1. Retrieve top-K FAISS results
    2. Rerank using cross-encoder
    3. Send best chunks to LLM
    """

    # Step 1: FAISS search
    retrieved = retrieve_top_k(query, k=top_k)

    # Step 2: Rerank
    reranked = rerank_chunks(query, retrieved)

    # Keep only the TOP chunk(s)
    best_chunks = reranked[:2]   # use top 2 chunks for context

    # Step 3: Build context for LLM
    context = build_context(best_chunks)

    # Step 4: Call offline LLM
    answer = call_llm_sync(query, context)

    return {
        "answer": answer,
        "sources": best_chunks
    }
