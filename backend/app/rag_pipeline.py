# rag_pipeline.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import List
from backend.app.retriever import retrieve_top_k
from backend.app.config import settings
import textwrap

def build_context(chunks: List[dict], max_chars: int = 4000) -> str:
    """
    Concatenate reranked chunks into a context string (with simple truncation).
    """
    parts = []
    total = 0
    for c in chunks:
        snippet = c.get("text", "").strip()
        entry = f"[Source: {c.get('document')} | Chunk: {c.get('chunk_id')}]\n{snippet}\n"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)

def answer_from_rag(query: str, top_k: int = None):
    if top_k is None:
        top_k = settings.TOP_K
    # 1) retrieve + rerank
    retrieved = retrieve_top_k(query, k=top_k)
    # 2) build context
    context = build_context(retrieved, max_chars=4000)
    # 3) construct prompt for LLM
    prompt = textwrap.dedent(f"""
    You are a helpful assistant. Use ONLY the context below to answer the question.
    If the answer is not present in the context, say: "The information is not available in the provided context."

    Context:
    {context}

    Question: {query}

    Answer:
    """).strip()

    # return prompt + sources so the caller (LLM early) can use either sync or streaming call.
    return {"prompt": prompt, "sources": retrieved}
