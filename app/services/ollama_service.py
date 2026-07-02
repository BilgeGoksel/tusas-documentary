"""Ollama connectivity helpers."""

import logging

import requests

from app.core.config import settings
from app.models.schemas import OllamaStatusResponse

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT_SECONDS = 3.0


def check_ollama_connection() -> OllamaStatusResponse:
    """Check whether the local Ollama service is reachable."""
    base_url = settings.ollama_base_url.rstrip("/")
    try:
        response = requests.get(
            f"{base_url}/api/tags",
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        if not response.ok:
            logger.warning("Ollama returned unhealthy status: %s", response.status_code)
        return OllamaStatusResponse(
            status="available" if response.ok else "unavailable",
            base_url=base_url,
        )
    except requests.RequestException as exc:
        logger.warning("Ollama connection failed: %s", exc)
        return OllamaStatusResponse(status="unavailable", base_url=base_url)
