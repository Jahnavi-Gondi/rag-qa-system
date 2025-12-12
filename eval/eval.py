import os
import sys
import json
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from backend.app.answer import answer_query
from sentence_transformers import SentenceTransformer, util

EVAL_FILE = "eval/sample_questions.json"
RESULTS_FILE = "eval/results.json"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(a, b):
    """Compute cosine similarity between answer and ground truth."""
    emb_a = embed_model.encode(a, convert_to_tensor=True)
    emb_b = embed_model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b)[0][0])

def evaluate():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)

    results = []
    start_time = time.time()

    for item in samples:
        question = item["question"]
        expected = item["answer"]

        print(f"\n[TEST] {question}")

        try:
            prediction = answer_query(question)["answer"]
        except Exception as e:
            prediction = f"ERROR: {str(e)}"

        sim = semantic_similarity(prediction, expected)

        results.append({
            "question": question,
            "expected": expected,
            "predicted": prediction,
            "semantic_similarity": sim
        })

        print(f"[Similarity] {sim:.4f}")

    total_time = time.time() - start_time

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "avg_similarity": sum(r["semantic_similarity"] for r in results) / len(results),
            "eval_time_sec": total_time
        }, f, indent=2)

    print("\n[DONE] Evaluation complete!")
    print(f"Average similarity: {results[-1]['semantic_similarity']:.4f}")
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    evaluate()
