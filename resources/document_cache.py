"""Download the source PDF and track its cached version."""

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from resources.config import METADATA_PATH, PDF_PATH, SOURCE_URL


CacheMetadata = dict[str, Any]


def load_cache_metadata() -> CacheMetadata:
    """Load information about the cached PDF and FAISS index."""
    if not METADATA_PATH.exists():
        return {}

    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache_metadata(metadata: CacheMetadata) -> None:
    """Save cache metadata atomically."""
    temporary_path = METADATA_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(METADATA_PATH)


def file_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_document(
    url: str,
    local_path: Path,
    metadata: CacheMetadata,
) -> tuple[bool, CacheMetadata]:
    """Synchronize a remote PDF and report whether its contents changed."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if local_path.exists() and metadata.get("source_url") == url:
        if metadata.get("etag"):
            headers["If-None-Match"] = metadata["etag"]
        if metadata.get("last_modified"):
            headers["If-Modified-Since"] = metadata["last_modified"]

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=60,
    ) as response:
        if response.status_code == requests.codes.not_modified:
            local_hash = file_sha256(local_path)
            document_changed = metadata.get("sha256") != local_hash
            metadata["sha256"] = local_hash
            save_cache_metadata(metadata)
            return document_changed, metadata

        response.raise_for_status()
        temporary_path = local_path.with_suffix(local_path.suffix + ".download")
        digest = hashlib.sha256()

        try:
            with temporary_path.open("wb") as file:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        digest.update(block)
                        file.write(block)

            document_hash = digest.hexdigest()
            document_changed = (
                not local_path.exists()
                or metadata.get("source_url") != url
                or metadata.get("sha256") != document_hash
            )

            if document_changed:
                temporary_path.replace(local_path)
            else:
                temporary_path.unlink()

            metadata.update(
                {
                    "source_url": url,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "sha256": document_hash,
                }
            )
            save_cache_metadata(metadata)
            return document_changed, metadata
        finally:
            temporary_path.unlink(missing_ok=True)


def ensure_document_current() -> tuple[bool, CacheMetadata]:
    """Update the local PDF, falling back to it when the network is unavailable."""
    metadata = load_cache_metadata()

    if not SOURCE_URL:
        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"PDF not found at {PDF_PATH}. Set PDF_PATH/PDF_FILENAME in .env "
                "or place your file under resources/files/."
            )
        local_hash = file_sha256(PDF_PATH)
        document_changed = metadata.get("sha256") != local_hash
        metadata.update({"source_url": None, "sha256": local_hash})
        save_cache_metadata(metadata)
        return document_changed, metadata

    try:
        return sync_document(SOURCE_URL, PDF_PATH, metadata)
    except requests.RequestException:
        if not PDF_PATH.exists():
            raise

        print("Could not check for a PDF update; using the local copy.")
        local_hash = file_sha256(PDF_PATH)
        document_changed = metadata.get("sha256") != local_hash
        metadata.update({"source_url": SOURCE_URL, "sha256": local_hash})
        save_cache_metadata(metadata)
        return document_changed, metadata
