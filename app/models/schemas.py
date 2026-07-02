"""Pydantic schemas used by the API."""

from datetime import datetime

from pydantic import BaseModel, Field


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


class ProcessedPageResponse(BaseModel):
    """Processed page content response."""

    page_number: int
    text: str
    extraction_method: str
    character_count: int
    requires_ocr: bool
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class DocumentProcessResponse(BaseModel):
    """Document processing response."""

    document_id: str
    original_filename: str
    page_count: int
    total_character_count: int
    processing_status: str
    processed_at: datetime
    from_cache: bool
    pages: list[ProcessedPageResponse]


class OllamaStatusResponse(BaseModel):
    """Ollama connection status response."""

    status: str
    base_url: str


class HealthResponse(BaseModel):
    """Health check response."""

    api_status: str
    ollama: OllamaStatusResponse
