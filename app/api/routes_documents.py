"""Document upload routes."""

from fastapi import APIRouter, File, UploadFile

from app.models.schemas import DocumentUploadResponse
from app.services.file_service import save_upload_file

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a supported document without extracting its contents."""
    return await save_upload_file(file)
