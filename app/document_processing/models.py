"""Models for extracted document content."""

from typing import Literal

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Text extracted from a single document page."""

    document_id: str
    filename: str
    page_number: int
    text: str
    extraction_method: Literal["native_pdf", "ocr", "ocr_pdf"] = "native_pdf"
    character_count: int
    requires_ocr: bool
    detected_language: str | None = None
    average_confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


PageContent = ExtractedPage
