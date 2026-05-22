"""
src/document_processor.py
──────────────────────────
Handles loading and chunking of uploaded documents.
Supports: PDF, DOCX, TXT
"""

import io
import tempfile
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from src.config import config


def load_document(uploaded_file) -> List[Document]:
    """
    Load a Streamlit UploadedFile into LangChain Documents.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        List of LangChain Document objects
    """
    file_name: str = uploaded_file.name
    file_bytes: bytes = uploaded_file.read()
    extension: str = os.path.splitext(file_name)[1].lower()

    # Write to a temp file so loaders can read from disk
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=extension
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif extension in (".docx", ".doc"):
            loader = Docx2txtLoader(tmp_path)
        elif extension == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                "Please upload a PDF, DOCX, or TXT file."
            )

        docs = loader.load()

        # Attach source metadata
        for doc in docs:
            doc.metadata["source"] = file_name

        return docs

    finally:
        os.unlink(tmp_path)


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks suitable for embedding.

    Args:
        documents: Raw loaded LangChain Documents

    Returns:
        Chunked LangChain Documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def process_uploaded_file(uploaded_file) -> List[Document]:
    """
    Full pipeline: load → split.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Chunked LangChain Documents ready for embedding
    """
    raw_docs = load_document(uploaded_file)
    chunks = split_documents(raw_docs)
    return chunks