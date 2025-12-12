# build_index.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


if __name__ == "__main__":
    # safe build script
    from backend.app.vectorstore import build_faiss_index
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hnsw", action="store_true", help="Use HNSW index (more memory but faster)")
    args = p.parse_args()
    build_faiss_index(use_hnsw=args.hnsw)
