"""Generate grounded answers through the local Ollama chat API."""

import logging
from time import perf_counter
from typing import Any

import requests

from app.core.config import settings
from app.rag.embedding_service import OllamaUnavailableError

logger = logging.getLogger(__name__)


class ChatGenerationError(Exception):
    """Base error raised when Ollama chat generation fails."""


class InvalidChatMessagesError(ChatGenerationError, ValueError):
    """Raised when chat messages are empty or malformed."""


class ChatModelNotFoundError(ChatGenerationError):
    """Raised when the configured Ollama chat model is unavailable."""


class InvalidChatResponseError(ChatGenerationError):
    """Raised when Ollama returns malformed or empty chat content."""


class ChatTimeoutError(OllamaUnavailableError):
    """Raised when Ollama chat generation exceeds its timeout."""


class ChatHTTPError(ChatGenerationError):
    """Raised when Ollama returns an unsuccessful chat HTTP response."""


def generate_answer(messages: list[dict[str, str]]) -> str:
    """Generate one non-streaming answer from validated chat messages."""
    _validate_messages(messages)
    model = settings.ollama_chat_model
    started_at = perf_counter()
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": settings.ollama_temperature},
    }

    try:
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=settings.ollama_chat_timeout_seconds,
        )
    except requests.Timeout as exc:
        elapsed = perf_counter() - started_at
        logger.warning(
            "Ollama chat timed out: model=%s elapsed_seconds=%.3f",
            model,
            elapsed,
        )
        raise ChatTimeoutError("Ollama chat istegi zaman asimina ugradi.") from exc
    except requests.RequestException as exc:
        elapsed = perf_counter() - started_at
        logger.warning(
            "Ollama chat unavailable: model=%s elapsed_seconds=%.3f error_type=%s",
            model,
            elapsed,
            type(exc).__name__,
        )
        raise OllamaUnavailableError("Ollama servisine baglanilamadi.") from exc

    elapsed = perf_counter() - started_at
    if response.status_code == 404:
        logger.warning("Ollama chat model not found: model=%s", model)
        raise ChatModelNotFoundError(f"Ollama chat modeli bulunamadi: {model}")
    if not response.ok:
        logger.warning(
            "Ollama chat failed: model=%s status_code=%s elapsed_seconds=%.3f",
            model,
            response.status_code,
            elapsed,
        )
        raise ChatHTTPError(
            f"Ollama chat istegi basarisiz oldu (HTTP {response.status_code})."
        )

    try:
        body = response.json()
    except (requests.JSONDecodeError, ValueError) as exc:
        raise InvalidChatResponseError(
            "Ollama gecersiz bir chat cevabi dondurdu."
        ) from exc

    content = _extract_content(body)
    logger.info(
        "Ollama chat completed: model=%s elapsed_seconds=%.3f",
        model,
        elapsed,
    )
    return content


def _validate_messages(messages: list[dict[str, str]]) -> None:
    if not isinstance(messages, list) or not messages:
        raise InvalidChatMessagesError("Chat messages listesi bos olamaz.")
    for message in messages:
        if not isinstance(message, dict):
            raise InvalidChatMessagesError("Chat message formati gecersiz.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise InvalidChatMessagesError("Chat message rolu gecersiz.")
        if not isinstance(content, str) or not content.strip():
            raise InvalidChatMessagesError("Chat message icerigi bos olamaz.")


def _extract_content(body: Any) -> str:
    if not isinstance(body, dict):
        raise InvalidChatResponseError("Chat cevabi bir nesne olmalidir.")
    message = body.get("message")
    if not isinstance(message, dict):
        raise InvalidChatResponseError("Chat cevabinda message alani bulunamadi.")
    content = message.get("content")
    if not isinstance(content, str):
        raise InvalidChatResponseError("Chat cevabinda content alani bulunamadi.")
    stripped_content = content.strip()
    if not stripped_content:
        raise InvalidChatResponseError("Ollama bos bir chat cevabi dondurdu.")
    return stripped_content


__all__ = [
    "ChatGenerationError",
    "ChatHTTPError",
    "ChatModelNotFoundError",
    "ChatTimeoutError",
    "InvalidChatMessagesError",
    "InvalidChatResponseError",
    "OllamaUnavailableError",
    "generate_answer",
]
