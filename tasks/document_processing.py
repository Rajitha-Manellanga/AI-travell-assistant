"""Load and split the source document."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from resources.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_document(file_path: Path):
    """Extract text from a PDF, returning one document per page."""
    return PyPDFLoader(str(file_path)).load()


def split_document(
    document,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """Split documents into chunks that do not exceed the configured size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(document)
