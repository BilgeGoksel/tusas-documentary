"""Models used by the document chunking pipeline."""

from typing import Literal

from pydantic import BaseModel


class Chunk(BaseModel):
    """A page-bounded text chunk with source metadata."""

    chunk_id: str
    document_id: str
    original_filename: str
    page_number: int
    chunk_index: int
    text: str
    character_count: int
    extraction_method: Literal["native_pdf", "ocr", "ocr_pdf"]
