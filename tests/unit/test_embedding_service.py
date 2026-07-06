"""Tests for local Ollama embedding generation."""

from unittest.mock import Mock

import pytest
import requests

from app.rag.embedding_service import (
    OLLAMA_EMBED_TIMEOUT_SECONDS,
    EmptyEmbeddingTextError,
    InvalidEmbeddingResponseError,
    OllamaHTTPError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    embed_text,
    embed_texts,
)


def make_response(status_code: int, body: object) -> Mock:
    """Create a requests-compatible mocked response."""
    response = Mock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    response.json.return_value = body
    return response


def test_embed_single_text(monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock(return_value=make_response(200, {"embeddings": [[0.1, 0.2]]}))
    monkeypatch.setattr("app.rag.embedding_service.requests.post", post)

    result = embed_text("Türkçe belge metni")

    assert result == [0.1, 0.2]
    post.assert_called_once_with(
        "http://localhost:11434/api/embed",
        json={"model": "qwen3-embedding:0.6b", "input": ["Türkçe belge metni"]},
        timeout=OLLAMA_EMBED_TIMEOUT_SECONDS,
    )


def test_embed_multiple_texts_in_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock(
        return_value=make_response(200, {"embeddings": [[1, 2], [3.0, 4.0]]})
    )
    monkeypatch.setattr("app.rag.embedding_service.requests.post", post)

    result = embed_texts(["ilk", "ikinci"])

    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert all(isinstance(value, float) for vector in result for value in vector)
    assert post.call_count == 1


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_rejected(text: str) -> None:
    with pytest.raises(EmptyEmbeddingTextError):
        embed_text(text)


def test_timeout_raises_specific_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.embedding_service.requests.post",
        Mock(side_effect=requests.Timeout("timeout")),
    )

    with pytest.raises(OllamaTimeoutError):
        embed_text("metin")


def test_connection_failure_raises_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.rag.embedding_service.requests.post",
        Mock(side_effect=requests.ConnectionError("connection refused")),
    )

    with pytest.raises(OllamaUnavailableError):
        embed_text("metin")


def test_404_raises_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.embedding_service.requests.post",
        Mock(return_value=make_response(404, {"error": "model not found"})),
    )

    with pytest.raises(OllamaModelNotFoundError):
        embed_text("metin")


def test_500_raises_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.embedding_service.requests.post",
        Mock(return_value=make_response(500, {"error": "internal error"})),
    )

    with pytest.raises(OllamaHTTPError, match="HTTP 500"):
        embed_text("metin")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"embeddings": []},
        {"embeddings": [[]]},
        {"embeddings": [[0.1, "gecersiz"]]},
        {"embeddings": [[True, 0.2]]},
        {"embeddings": [[float("nan"), 0.2]]},
    ],
)
def test_invalid_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch, body: object
) -> None:
    monkeypatch.setattr(
        "app.rag.embedding_service.requests.post",
        Mock(return_value=make_response(200, body)),
    )

    with pytest.raises(InvalidEmbeddingResponseError):
        embed_text("metin")


def test_batch_response_count_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.rag.embedding_service.requests.post",
        Mock(return_value=make_response(200, {"embeddings": [[0.1, 0.2]]})),
    )

    with pytest.raises(InvalidEmbeddingResponseError, match="uyusmuyor"):
        embed_texts(["ilk", "ikinci"])
