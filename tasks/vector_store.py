"""Create, persist, and restore the FAISS vector store."""

from langchain_community.vectorstores import FAISS

from resources.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_PATH,
    PDF_PATH,
)
from resources.document_cache import CacheMetadata, save_cache_metadata
from resources.model_api import create_embedding_model
from tasks.document_processing import load_document, split_document


def index_is_current(metadata: CacheMetadata) -> bool:
    """Return whether the saved local index matches the PDF and settings."""
    return (
        (INDEX_PATH / "index.faiss").exists()
        and (INDEX_PATH / "index.pkl").exists()
        and metadata.get("indexed_sha256") == metadata.get("sha256")
        and metadata.get("embedding_model") == EMBEDDING_MODEL
        and metadata.get("chunk_size") == CHUNK_SIZE
        and metadata.get("chunk_overlap") == CHUNK_OVERLAP
    )


def get_vector_store(document_changed: bool, metadata: CacheMetadata):
    """Load the local FAISS index or rebuild it from the current PDF."""
    embeddings = create_embedding_model()

    if not document_changed and index_is_current(metadata):
        print("Loading the existing local FAISS index.")
        return FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("Building the local FAISS index from the current PDF.")
    documents = load_document(PDF_PATH)
    chunks = split_document(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_PATH))

    metadata.update(
        {
            "indexed_sha256": metadata.get("sha256"),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
    )
    save_cache_metadata(metadata)
    return vector_store
