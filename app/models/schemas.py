"""Pydantic schemas used by the API."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


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


class DocumentIndexResponse(BaseModel):
    """Document vector indexing response and cached summary."""

    document_id: str
    original_filename: str
    page_count: int
    chunk_count: int
    embedding_model: str
    indexing_status: str
    from_cache: bool
    indexed_at: datetime


class SearchRequest(BaseModel):
    """Semantic chunk search request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    document_ids: list[str] | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject queries containing only whitespace."""
        if not value.strip():
            raise ValueError("Arama sorgusu bos olamaz.")
        return value


class SearchResultResponse(BaseModel):
    """One relevant document chunk returned by semantic search."""

    chunk_id: str
    text: str
    document_id: str
    original_filename: str
    page_number: int
    chunk_index: int
    extraction_method: str
    similarity_score: float


class SearchResponse(BaseModel):
    """Semantic chunk search response."""

    query: str
    result_count: int
    results: list[SearchResultResponse]


class QASourceResponse(BaseModel):
    """Source metadata for one chunk used to generate an answer."""

    source_number: int
    document_id: str
    original_filename: str
    page_number: int
    chunk_id: str
    similarity_score: float
    snippet: str


class QAResponse(BaseModel):
    """Grounded question-answering service result."""

    answer: str
    found_in_documents: bool
    sources: list[QASourceResponse]
    retrieved_chunk_count: int
    model: str
    top_k: int


class QARequest(SearchRequest):
    """Grounded question-answering request."""

    top_k: int = Field(default=6, ge=1, le=20, strict=True)


class OllamaStatusResponse(BaseModel):
    """Ollama connection status response."""

    status: str
    base_url: str


class HealthResponse(BaseModel):
    """Health check response."""

    api_status: str
    ollama: OllamaStatusResponse
