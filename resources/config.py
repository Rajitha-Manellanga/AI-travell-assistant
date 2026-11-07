"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RESOURCE_FILES_DIR = PROJECT_ROOT / "resources" / "files"

PDF_FILENAME = os.getenv("PDF_FILENAME", "example_eticket.pdf")
PDF_PATH = Path(os.getenv("PDF_PATH", str(RESOURCE_FILES_DIR / PDF_FILENAME)))
SOURCE_URL = os.getenv("SOURCE_URL", "").strip() or None
METADATA_PATH = PROJECT_ROOT / "document_cache.json"
INDEX_PATH = PROJECT_ROOT / "faiss_index"

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "ollama_cloud").lower()
OLLAMA_CLOUD_URL = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL") or os.getenv(
    "LLAMA_API_MODEL"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVER_DOCUMENT_COUNT = 6
MAX_HISTORY_MESSAGES = 12
