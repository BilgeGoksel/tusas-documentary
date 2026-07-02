"""Health endpoint tests."""

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import OllamaStatusResponse

client = TestClient(app)


def test_health_returns_available_ollama(monkeypatch) -> None:
    """Health endpoint reports available Ollama when service check succeeds."""

    def mock_check_ollama_connection() -> OllamaStatusResponse:
        return OllamaStatusResponse(
            status="available",
            base_url="http://localhost:11434",
        )

    monkeypatch.setattr(
        "app.api.routes_health.check_ollama_connection",
        mock_check_ollama_connection,
    )

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "api_status": "ok",
        "ollama": {
            "status": "available",
            "base_url": "http://localhost:11434",
        },
    }


def test_health_returns_200_when_ollama_unavailable(monkeypatch) -> None:
    """Health endpoint stays available when Ollama service check fails."""

    def mock_check_ollama_connection() -> OllamaStatusResponse:
        return OllamaStatusResponse(
            status="unavailable",
            base_url="http://localhost:11434",
        )

    monkeypatch.setattr(
        "app.api.routes_health.check_ollama_connection",
        mock_check_ollama_connection,
    )

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_status"] == "ok"
    assert payload["ollama"]["status"] == "unavailable"
