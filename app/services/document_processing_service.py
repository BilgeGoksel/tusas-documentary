"""Service layer for processing uploaded documents."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.document_processing.document_processor import (
    UnsupportedDocumentFormatError,
    process_document,
)
from app.document_processing.image_preprocessor import ImagePreprocessingError
from app.document_processing.ocr_service import OCRServiceError
from app.document_processing.pdf_extractor import (
    EncryptedPDFError,
    InvalidPDFError,
    PDFExtractionError,
)
from app.models.schemas import DocumentProcessResponse, ProcessedPageResponse
from app.services.file_service import METADATA_FILENAME


class DocumentProcessingServiceError(Exception):
    """Base error for uploaded document processing."""


class DocumentMetadataNotFoundError(DocumentProcessingServiceError):
    """Raised when the document id is not present in metadata."""


class StoredDocumentMissingError(DocumentProcessingServiceError):
    """Raised when the stored file is missing from disk."""


class ProcessedResultStorageError(DocumentProcessingServiceError):
    """Raised when a cached processing result cannot be read or written."""


def process_uploaded_document(
    document_id: str,
    force: bool = False,
) -> DocumentProcessResponse:
    """Process an uploaded document by id, using cached results when available."""
    processed_path = _processed_result_path(document_id)
    if processed_path.exists() and not force:
        return _load_processed_result(processed_path)

    record = _find_document_record(document_id)
    stored_file = Path(settings.upload_dir) / str(record["stored_filename"])
    if not stored_file.is_file():
        raise StoredDocumentMissingError("Kayitli belge dosyasi diskte bulunamadi.")

    pages = process_document(
        stored_file,
        document_id=str(record["document_id"]),
        original_filename=str(record["original_filename"]),
        content_type=str(record["content_type"]),
    )
    response = _build_processing_response(
        document_id=str(record["document_id"]),
        original_filename=str(record["original_filename"]),
        pages=[
            ProcessedPageResponse(
                page_number=page.page_number,
                text=page.text,
                extraction_method=page.extraction_method,
                character_count=page.character_count,
                requires_ocr=page.requires_ocr,
                confidence=page.average_confidence,
                warnings=page.warnings,
            )
            for page in pages
        ],
        from_cache=False,
    )
    _write_processed_result(processed_path, response)
    return response


def _find_document_record(document_id: str) -> dict[str, Any]:
    metadata_path = Path(settings.upload_dir) / METADATA_FILENAME
    if not metadata_path.exists():
        raise DocumentMetadataNotFoundError("Belge bulunamadi.")

    try:
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ProcessedResultStorageError("Belge metadata kaydi okunamadi.") from exc

    if not isinstance(records, list):
        raise ProcessedResultStorageError("Belge metadata kaydi gecersiz.")

    for record in records:
        if isinstance(record, dict) and str(record.get("document_id")) == document_id:
            return record
    raise DocumentMetadataNotFoundError("Belge bulunamadi.")


def _processed_result_path(document_id: str) -> Path:
    return Path(settings.processed_dir) / f"{document_id}.json"


def _load_processed_result(path: Path) -> DocumentProcessResponse:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["from_cache"] = True
        return DocumentProcessResponse.model_validate(payload)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ProcessedResultStorageError("Isleme sonucu okunamadi.") from exc


def _write_processed_result(path: Path, response: DocumentProcessResponse) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    try:
        temp_path.write_text(
            json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise ProcessedResultStorageError("Isleme sonucu kaydedilemedi.") from exc


def _build_processing_response(
    document_id: str,
    original_filename: str,
    pages: list[ProcessedPageResponse],
    from_cache: bool,
) -> DocumentProcessResponse:
    return DocumentProcessResponse(
        document_id=document_id,
        original_filename=original_filename,
        page_count=len(pages),
        total_character_count=sum(page.character_count for page in pages),
        processing_status="processed",
        processed_at=datetime.now(timezone.utc),
        from_cache=from_cache,
        pages=pages,
    )


PROCESSING_CLIENT_ERRORS = (
    EncryptedPDFError,
    InvalidPDFError,
    PDFExtractionError,
    ImagePreprocessingError,
)


PROCESSING_SERVER_ERRORS = (
    OCRServiceError,
)


__all__ = [
    "DocumentMetadataNotFoundError",
    "DocumentProcessingServiceError",
    "PROCESSING_CLIENT_ERRORS",
    "PROCESSING_SERVER_ERRORS",
    "ProcessedResultStorageError",
    "StoredDocumentMissingError",
    "UnsupportedDocumentFormatError",
    "process_uploaded_document",
]
