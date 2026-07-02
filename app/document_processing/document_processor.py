"""Hybrid document processing entry point."""

from pathlib import Path
from typing import Any

import fitz
import numpy as np

from app.document_processing.models import PageContent
from app.document_processing.ocr_service import (
    OCR_PDF_EXTRACTION_METHOD,
    extract_image_array_text,
    extract_image_text,
)
from app.document_processing.pdf_extractor import extract_pdf_pages

PDF_CONTENT_TYPE = "application/pdf"
IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png"}
PDF_RENDER_ZOOM = 2.0


class DocumentProcessingError(Exception):
    """Raised when a document cannot be processed safely."""


class UnsupportedDocumentFormatError(DocumentProcessingError):
    """Raised when the document format is not supported."""


class PDFRenderError(DocumentProcessingError):
    """Raised when a PDF page cannot be rendered for OCR."""


def process_document(
    file_path: str | Path,
    document_id: str,
    original_filename: str,
    content_type: str,
    ocr_engine: Any | None = None,
) -> list[PageContent]:
    """Process a supported document and return ordered page content."""
    if content_type == PDF_CONTENT_TYPE:
        return _process_pdf(file_path, document_id, original_filename, ocr_engine)
    if content_type in IMAGE_CONTENT_TYPES:
        return [
            extract_image_text(
                file_path,
                document_id=document_id,
                filename=original_filename,
                ocr_engine=ocr_engine,
            )
        ]

    raise UnsupportedDocumentFormatError("Desteklenmeyen belge formati.")


def _process_pdf(
    file_path: str | Path,
    document_id: str,
    original_filename: str,
    ocr_engine: Any | None,
) -> list[PageContent]:
    native_pages = extract_pdf_pages(
        file_path,
        document_id=document_id,
        filename=original_filename,
    )
    if not any(page.requires_ocr for page in native_pages):
        return native_pages

    path = Path(file_path)
    try:
        document = fitz.open(path)
    except (fitz.EmptyFileError, fitz.FileDataError, RuntimeError) as exc:
        raise PDFRenderError("PDF OCR icin render edilemedi.") from exc

    try:
        processed_pages: list[PageContent] = []
        for native_page in native_pages:
            if not native_page.requires_ocr:
                processed_pages.append(native_page)
                continue

            image = _render_pdf_page(document, native_page.page_number)
            processed_pages.append(
                extract_image_array_text(
                    image=image,
                    document_id=document_id,
                    filename=original_filename,
                    page_number=native_page.page_number,
                    extraction_method=OCR_PDF_EXTRACTION_METHOD,
                    ocr_engine=ocr_engine,
                )
            )
        return processed_pages
    finally:
        document.close()


def _render_pdf_page(document: fitz.Document, page_number: int) -> np.ndarray:
    page_index = page_number - 1
    try:
        page = document.load_page(page_index)
        matrix = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    except (RuntimeError, ValueError) as exc:
        raise PDFRenderError("PDF sayfasi OCR icin goruntuye donusturulemedi.") from exc

    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )
    if pixmap.n == 1:
        return np.repeat(image, 3, axis=2)
    if pixmap.n > 3:
        return image[:, :, :3]
    return image
