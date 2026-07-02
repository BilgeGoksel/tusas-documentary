"""Pydantic schemas used by the API."""

from datetime import datetime

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    """Document upload response."""

    document_id: str
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    is_duplicate: bool
    status: str
    created_at: datetime


class OllamaStatusResponse(BaseModel):
    """Ollama connection status response."""

    status: str
    base_url: str


class HealthResponse(BaseModel):
    """Health check response."""

    api_status: str
    ollama: OllamaStatusResponse
