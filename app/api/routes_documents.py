"""Document upload routes."""

import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.models.schemas import DocumentProcessResponse, DocumentUploadResponse
from app.services.document_processing_service import (
    PROCESSING_CLIENT_ERRORS,
    PROCESSING_SERVER_ERRORS,
    DocumentMetadataNotFoundError,
    ProcessedResultStorageError,
    StoredDocumentMissingError,
    UnsupportedDocumentFormatError,
    process_uploaded_document,
)
from app.services.file_service import save_upload_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a supported document without extracting its contents."""
    return await save_upload_file(file)


@router.post("/{document_id}/process", response_model=DocumentProcessResponse)
def process_document_endpoint(
    document_id: str,
    force: bool = Query(default=False),
) -> DocumentProcessResponse:
    """Process a previously uploaded document by id."""
    try:
        return process_uploaded_document(document_id=document_id, force=force)
    except DocumentMetadataNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Belge bulunamadi.",
        ) from exc
    except StoredDocumentMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Kayitli belge dosyasi diskte bulunamadi.",
        ) from exc
    except UnsupportedDocumentFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Desteklenmeyen belge formati.",
        ) from exc
    except PROCESSING_CLIENT_ERRORS as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except PROCESSING_SERVER_ERRORS + (ProcessedResultStorageError,) as exc:
        logger.exception("Document processing failed for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Belge islenirken bir hata olustu.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected document processing error for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Belge islenirken bir hata olustu.",
        ) from exc
