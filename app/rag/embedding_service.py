"""Generate text embeddings through the local Ollama API."""

import logging
import math
from typing import Any

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

OLLAMA_EMBED_TIMEOUT_SECONDS = 30.0


class EmbeddingServiceError(Exception):
    """Base error raised when embedding generation fails."""


class EmptyEmbeddingTextError(EmbeddingServiceError, ValueError):
    """Raised when an embedding input is empty or whitespace-only."""


class OllamaUnavailableError(EmbeddingServiceError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaTimeoutError(OllamaUnavailableError):
    """Raised when Ollama does not respond within the configured timeout."""


class OllamaModelNotFoundError(EmbeddingServiceError):
    """Raised when Ollama does not have the configured embedding model."""


class OllamaHTTPError(EmbeddingServiceError):
    """Raised when Ollama returns an unsuccessful HTTP response."""


class InvalidEmbeddingResponseError(EmbeddingServiceError):
    """Raised when Ollama returns an invalid or inconsistent response."""


def embed_text(text: str) -> list[float]:
    """Generate an embedding for one non-empty text value."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a non-empty batch of text values."""
    _validate_texts(texts)
    payload = {
        "model": settings.ollama_embedding_model,
        "input": texts,
    }

    try:
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/embed",
            json=payload,
            timeout=OLLAMA_EMBED_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        logger.warning("Ollama embedding request timed out.")
        raise OllamaTimeoutError("Ollama embedding istegi zaman asimina ugradi.") from exc
    except requests.RequestException as exc:
        logger.warning("Ollama embedding service is unavailable: %s", type(exc).__name__)
        raise OllamaUnavailableError("Ollama servisine baglanilamadi.") from exc

    if response.status_code == 404:
        logger.warning("Ollama embedding model was not found.")
        raise OllamaModelNotFoundError(
            f"Ollama embedding modeli bulunamadi: {settings.ollama_embedding_model}"
        )
    if not response.ok:
        logger.warning(
            "Ollama embedding request failed with HTTP %s.", response.status_code
        )
        raise OllamaHTTPError(
            f"Ollama embedding istegi basarisiz oldu (HTTP {response.status_code})."
        )

    try:
        body = response.json()
    except (requests.JSONDecodeError, ValueError) as exc:
        raise InvalidEmbeddingResponseError(
            "Ollama gecersiz bir embedding cevabi dondurdu."
        ) from exc
    return _validate_embeddings(body, expected_count=len(texts))


def _validate_texts(texts: list[str]) -> None:
    if not texts:
        raise EmptyEmbeddingTextError("Embedding icin en az bir metin gereklidir.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise EmptyEmbeddingTextError("Embedding metni bos olamaz.")


def _validate_embeddings(body: Any, expected_count: int) -> list[list[float]]:
    if not isinstance(body, dict):
        raise InvalidEmbeddingResponseError("Embedding cevabi bir nesne olmalidir.")

    embeddings = body.get("embeddings")
    if not isinstance(embeddings, list):
        raise InvalidEmbeddingResponseError("Cevapta embeddings listesi bulunamadi.")
    if len(embeddings) != expected_count:
        raise InvalidEmbeddingResponseError(
            "Embedding cevap sayisi girdi sayisiyla uyusmuyor."
        )

    validated: list[list[float]] = []
    vector_size: int | None = None
    for embedding in embeddings:
        if not isinstance(embedding, list) or not embedding:
            raise InvalidEmbeddingResponseError("Embedding vektoru bos veya gecersiz.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in embedding
        ):
            raise InvalidEmbeddingResponseError(
                "Embedding vektoru yalnizca sayisal degerler icermelidir."
            )

        vector = [float(value) for value in embedding]
        if any(not math.isfinite(value) for value in vector):
            raise InvalidEmbeddingResponseError(
                "Embedding vektoru sonlu olmayan degerler iceremez."
            )
        if vector_size is None:
            vector_size = len(vector)
        elif len(vector) != vector_size:
            raise InvalidEmbeddingResponseError(
                "Batch embedding vektor boyutlari birbiriyle uyusmuyor."
            )
        validated.append(vector)
    return validated


__all__ = [
    "EmbeddingServiceError",
    "EmptyEmbeddingTextError",
    "InvalidEmbeddingResponseError",
    "OllamaHTTPError",
    "OllamaModelNotFoundError",
    "OllamaTimeoutError",
    "OllamaUnavailableError",
    "embed_text",
    "embed_texts",
]
