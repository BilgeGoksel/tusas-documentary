"""Native text extraction for text-based PDF documents."""

from pathlib import Path

import fitz

from app.core.config import settings
from app.document_processing.models import ExtractedPage
from app.document_processing.text_cleaner import clean_extracted_text

NATIVE_PDF_EXTRACTION_METHOD = "native_pdf"


class PDFExtractionError(Exception):
    """Raised when PDF text extraction cannot be completed safely."""


class EncryptedPDFError(PDFExtractionError):
    """Raised when a PDF requires a password."""


class InvalidPDFError(PDFExtractionError):
    """Raised when a PDF cannot be opened as a valid document."""


def extract_pdf_pages(
    pdf_path: str | Path,
    document_id: str,
    filename: str,
) -> list[ExtractedPage]:
    """Extract page-level native text from a text-based PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise InvalidPDFError("PDF dosyasi bulunamadi.")

    try:
        document = fitz.open(path)
    except (fitz.EmptyFileError, fitz.FileDataError, RuntimeError) as exc:
        raise InvalidPDFError("PDF dosyasi okunamadi veya bozuk.") from exc

    try:
        if document.needs_pass:
            raise EncryptedPDFError("Sifreli PDF dosyalari henuz desteklenmiyor.")

        pages: list[ExtractedPage] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            cleaned_text = clean_extracted_text(page.get_text("text"))
            character_count = len(cleaned_text)
            pages.append(
                ExtractedPage(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_index + 1,
                    text=cleaned_text,
                    extraction_method=NATIVE_PDF_EXTRACTION_METHOD,
                    character_count=character_count,
                    requires_ocr=character_count < settings.pdf_ocr_min_text_chars,
                )
            )
        return pages
    finally:
        document.close()
