"""Tests for local Ollama grounded answer generation."""

from unittest.mock import Mock

import pytest
import requests

from app.rag.answer_generator import (
    ChatHTTPError,
    ChatModelNotFoundError,
    ChatTimeoutError,
    InvalidChatMessagesError,
    InvalidChatResponseError,
    OllamaUnavailableError,
    generate_answer,
)

MESSAGES = [
    {"role": "system", "content": "Yalnızca belgeyi kullan."},
    {"role": "user", "content": "Belge bağlamı"},
    {"role": "user", "content": "Soru nedir?"},
]


def make_response(status_code: int, body: object) -> Mock:
    """Create a requests-compatible mocked response."""
    response = Mock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    response.json.return_value = body
    return response


def test_generate_answer_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.answer_generator.requests.post",
        Mock(return_value=make_response(200, {"message": {"content": "  Yanıt [1]  "}})),
    )

    assert generate_answer(MESSAGES) == "Yanıt [1]"


def test_ollama_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.answer_generator.requests.post",
        Mock(side_effect=requests.ConnectionError("refused")),
    )

    with pytest.raises(OllamaUnavailableError):
        generate_answer(MESSAGES)


def test_chat_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.answer_generator.requests.post",
        Mock(side_effect=requests.Timeout("timeout")),
    )

    with pytest.raises(ChatTimeoutError):
        generate_answer(MESSAGES)


def test_chat_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.answer_generator.requests.post",
        Mock(return_value=make_response(404, {"error": "model not found"})),
    )

    with pytest.raises(ChatModelNotFoundError, match="qwen3:4b"):
        generate_answer(MESSAGES)


def test_chat_500_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.answer_generator.requests.post",
        Mock(return_value=make_response(500, {"error": "internal"})),
    )

    with pytest.raises(ChatHTTPError, match="HTTP 500"):
        generate_answer(MESSAGES)


def test_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    response = make_response(200, {})
    response.json.side_effect = requests.JSONDecodeError("bad json", "", 0)
    monkeypatch.setattr("app.rag.answer_generator.requests.post", Mock(return_value=response))

    with pytest.raises(InvalidChatResponseError):
        generate_answer(MESSAGES)


@pytest.mark.parametrize("body", [{}, {"message": {}}, {"message": {"content": None}}])
def test_missing_message_content(
    monkeypatch: pytest.MonkeyPatch, body: object
) -> None:
    monkeypatch.setattr(
        "app.rag.answer_generator.requests.post",
        Mock(return_value=make_response(200, body)),
    )

    with pytest.raises(InvalidChatResponseError):
        generate_answer(MESSAGES)


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_empty_answer(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    monkeypatch.setattr(
        "app.rag.answer_generator.requests.post",
        Mock(return_value=make_response(200, {"message": {"content": content}})),
    )

    with pytest.raises(InvalidChatResponseError):
        generate_answer(MESSAGES)


def test_request_uses_configured_model_temperature_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(return_value=make_response(200, {"message": {"content": "Yanıt"}}))
    monkeypatch.setattr("app.rag.answer_generator.requests.post", post)

    generate_answer(MESSAGES)

    post.assert_called_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:4b",
            "messages": MESSAGES,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120.0,
    )


def test_empty_messages_are_rejected_before_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock()
    monkeypatch.setattr("app.rag.answer_generator.requests.post", post)

    with pytest.raises(InvalidChatMessagesError):
        generate_answer([])

    post.assert_not_called()
