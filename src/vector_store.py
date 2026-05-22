from typing import List, Optional

from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from src.config import config


# Build Vector Store
def build_vector_store(
    chunks: List[Document]
) -> FAISS:

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


# Retriever
def get_retriever(
    vector_store: FAISS,
    k: Optional[int] = None
):

    k = k or config.RETRIEVAL_K

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    return retriever