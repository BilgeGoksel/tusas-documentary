"""Orchestrate processed-document indexing into the vector store."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.document_processing.models import PageContent
from app.models.schemas import DocumentIndexResponse, DocumentProcessResponse
from app.rag.chunker import chunk_pages
from app.rag.vector_store import add_chunks
from app.services.file_service import METADATA_FILENAME


class IndexingServiceError(Exception):
    """Base error raised by document indexing orchestration."""


class IndexDocumentNotFoundError(IndexingServiceError):
    """Raised when an uploaded document metadata record does not exist."""


class DocumentNotProcessedError(IndexingServiceError):
    """Raised when a document has no stored processing result."""


class EmptyProcessedDocumentError(IndexingServiceError):
    """Raised when a processed document contains no indexable text."""


class IndexingResultStorageError(IndexingServiceError):
    """Raised when processing or indexing metadata cannot be read or written."""


def index_processed_document(
    document_id: str,
    force: bool = False,
) -> DocumentIndexResponse:
    """Index a processed document, returning its cached summary when available."""
    record = _find_document_record(document_id)
    indexed_path = _indexed_result_path(document_id)
    if indexed_path.exists() and not force:
        return _load_indexed_result(indexed_path)

    processed = _load_processed_document(document_id)
    if processed.document_id != document_id:
        raise IndexingResultStorageError("Islenmis belge kimligi kayitla uyusmuyor.")
    original_filename = str(record["original_filename"])
    pages = _to_page_content(processed, original_filename)
    chunks = chunk_pages(pages)
    if not chunks:
        raise EmptyProcessedDocumentError(
            "Islenmis belgede indexlenebilir metin bulunamadi."
        )

    add_chunks(chunks)
    response = DocumentIndexResponse(
        document_id=document_id,
        original_filename=original_filename,
        page_count=len(pages),
        chunk_count=len(chunks),
        embedding_model=settings.ollama_embedding_model,
        indexing_status="indexed",
        from_cache=False,
        indexed_at=datetime.now(timezone.utc),
    )
    _write_indexed_result(indexed_path, response)
    return response


def _find_document_record(document_id: str) -> dict[str, Any]:
    metadata_path = Path(settings.upload_dir) / METADATA_FILENAME
    if not metadata_path.exists():
        raise IndexDocumentNotFoundError("Belge bulunamadi.")
    try:
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise IndexingResultStorageError("Belge metadata kaydi okunamadi.") from exc
    if not isinstance(records, list):
        raise IndexingResultStorageError("Belge metadata kaydi gecersiz.")

    for record in records:
        if isinstance(record, dict) and str(record.get("document_id")) == document_id:
            if "original_filename" not in record:
                raise IndexingResultStorageError("Belge metadata kaydi gecersiz.")
            return record
    raise IndexDocumentNotFoundError("Belge bulunamadi.")


def _load_processed_document(document_id: str) -> DocumentProcessResponse:
    path = Path(settings.processed_dir) / f"{document_id}.json"
    if not path.is_file():
        raise DocumentNotProcessedError("Belge henuz islenmemis.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DocumentProcessResponse.model_validate(payload)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise IndexingResultStorageError("Islenmis belge kaydi okunamadi.") from exc


def _to_page_content(
    processed: DocumentProcessResponse,
    original_filename: str,
) -> list[PageContent]:
    try:
        return [
            PageContent(
                document_id=processed.document_id,
                filename=original_filename,
                page_number=page.page_number,
                text=page.text,
                extraction_method=page.extraction_method,
                character_count=page.character_count,
                requires_ocr=page.requires_ocr,
                average_confidence=page.confidence,
                warnings=page.warnings,
            )
            for page in processed.pages
        ]
    except ValueError as exc:
        raise IndexingResultStorageError("Islenmis belge sayfa kaydi gecersiz.") from exc


def _indexed_result_path(document_id: str) -> Path:
    return Path(settings.indexed_dir) / f"{document_id}.json"


def _load_indexed_result(path: Path) -> DocumentIndexResponse:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["from_cache"] = True
        return DocumentIndexResponse.model_validate(payload)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise IndexingResultStorageError("Indexleme sonucu okunamadi.") from exc


def _write_indexed_result(path: Path, response: DocumentIndexResponse) -> None:
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
        raise IndexingResultStorageError("Indexleme sonucu kaydedilemedi.") from exc


__all__ = [
    "DocumentNotProcessedError",
    "EmptyProcessedDocumentError",
    "IndexDocumentNotFoundError",
    "IndexingResultStorageError",
    "IndexingServiceError",
    "index_processed_document",
]
