"""Document processing utilities."""

from app.document_processing.document_processor import (
    DocumentProcessingError,
    UnsupportedDocumentFormatError,
    process_document,
)
from app.document_processing.models import ExtractedPage
from app.document_processing.ocr_service import (
    OCRServiceError,
    OCRUnavailableError,
    extract_image_text,
)
from app.document_processing.pdf_extractor import (
    EncryptedPDFError,
    InvalidPDFError,
    PDFExtractionError,
    extract_pdf_pages,
)

__all__ = [
    "DocumentProcessingError",
    "EncryptedPDFError",
    "ExtractedPage",
    "InvalidPDFError",
    "OCRServiceError",
    "OCRUnavailableError",
    "PDFExtractionError",
    "UnsupportedDocumentFormatError",
    "extract_image_text",
    "extract_pdf_pages",
    "process_document",
]
