import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 300))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # Retrieval settings
    TOP_K: int = int(os.getenv("TOP_K", 8))

settings = Settings()
