import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pdfplumber
import json
from backend.app.config import settings

DATA_DIR = "data/docs"
CHUNKS_DIR = "data/chunks"


def load_pdf(filepath):
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"
    return text


def chunk_text(text, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def ingest_documents():
    """Load all documents from data/docs/, chunk them, and store into data/chunks/."""
    all_chunks = []
    file_id = 0

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)

        if filename.lower().endswith(".pdf"):
            text = load_pdf(filepath)
        else:
            # Assume plain text for now
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        chunks = chunk_text(text)
        print(f"[INFO] {filename}: {len(chunks)} chunks created.")

        # Save chunks
        for i, chunk in enumerate(chunks):
            chunk_obj = {
                "id": f"doc{file_id}_chunk{i}",
                "document": filename,
                "text": chunk
            }

            # Save as JSON file
            out_path = os.path.join(CHUNKS_DIR, f"doc{file_id}_chunk{i}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunk_obj, f, indent=2)

            all_chunks.append(chunk_obj)

        file_id += 1

    print(f"[DONE] Total chunks saved: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    ingest_documents()
