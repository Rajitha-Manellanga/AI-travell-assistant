"""Local and Ollama Cloud model clients."""

import os

from langchain_ollama import ChatOllama, OllamaEmbeddings

from resources.config import (
    CHAT_MODEL,
    CHAT_PROVIDER,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_CLOUD_MODEL,
    OLLAMA_CLOUD_URL,
)

def _create_ollama_cloud_model(
    temperature: float,
    max_tokens: int,
) -> ChatOllama:
    """Create an authenticated client for Ollama's hosted API."""
    # LLAMA_API_KEY is accepted temporarily to ease migration from the old config.
    api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("LLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY must be set when CHAT_PROVIDER=ollama_cloud."
        )
    if not OLLAMA_CLOUD_MODEL:
        raise RuntimeError(
            "OLLAMA_CLOUD_MODEL must be set to a model available to your "
            "Ollama account."
        )

    return ChatOllama(
        base_url=OLLAMA_CLOUD_URL,
        model=OLLAMA_CLOUD_MODEL,
        temperature=temperature,
        num_predict=max_tokens,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {api_key}"},
        },
    )


def create_chat_model(
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    """Create either a local Ollama model or an Ollama Cloud model."""
    if CHAT_PROVIDER == "ollama":
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=CHAT_MODEL,
            temperature=temperature,
            num_predict=max_tokens,
        )

    if CHAT_PROVIDER in {"ollama_cloud", "llama_api"}:
        return _create_ollama_cloud_model(temperature, max_tokens)

    raise ValueError(
        f"Unsupported CHAT_PROVIDER={CHAT_PROVIDER!r}. "
        "Use 'ollama' or 'ollama_cloud'."
    )


def create_embedding_model(
    model: str = EMBEDDING_MODEL,
) -> OllamaEmbeddings:
    """Create the local embedding client used by FAISS retrieval."""
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=model,
    )
