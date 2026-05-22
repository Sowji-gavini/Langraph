import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # ─────────────────────────────────────────
    # GROQ
    # ─────────────────────────────────────────

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    GROQ_MODEL = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    # ─────────────────────────────────────────
    # EMBEDDINGS
    # ─────────────────────────────────────────

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ─────────────────────────────────────────
    # WHISPER
    # ─────────────────────────────────────────

    WHISPER_MODEL = os.getenv(
        "WHISPER_MODEL",
        "whisper-large-v3"
    )

    # ─────────────────────────────────────────
    # CHUNKING
    # ─────────────────────────────────────────

    CHUNK_SIZE = int(
        os.getenv("CHUNK_SIZE", 1000)
    )

    CHUNK_OVERLAP = int(
        os.getenv("CHUNK_OVERLAP", 200)
    )

    # ─────────────────────────────────────────
    # RETRIEVAL
    # ─────────────────────────────────────────

    RETRIEVAL_K = int(
        os.getenv("RETRIEVAL_K", 4)
    )


config = Config()