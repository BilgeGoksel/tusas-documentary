"""Health and local service check routes."""

from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.ollama_service import check_ollama_connection

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return API and local Ollama health status."""
    return HealthResponse(api_status="ok", ollama=check_ollama_connection())
