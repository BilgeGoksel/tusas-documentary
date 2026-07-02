"""File upload validation and storage helpers."""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.models.schemas import DocumentUploadResponse

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
}
CHUNK_SIZE_BYTES = 1024 * 1024
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
METADATA_FILENAME = "documents.json"
_METADATA_LOCK = Lock()


def sanitize_filename(filename: str | None) -> str:
    """Return a safe filename without path components."""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dosya adi bulunamadi.",
        )

    name = Path(filename.replace("\\", "/")).name
    sanitized = SAFE_FILENAME_PATTERN.sub("_", name).strip("._")
    if not sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gecerli bir dosya adi bulunamadi.",
        )
    return sanitized


def validate_extension(filename: str) -> str:
    """Validate and return the file extension."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Yalnizca PDF, JPG, JPEG ve PNG dosyalari yuklenebilir.",
        )
    return extension


def validate_content_type(content_type: str | None) -> str:
    """Validate and return the uploaded file MIME type."""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Desteklenmeyen dosya turu.",
        )
    return content_type


def get_metadata_path(upload_dir: Path) -> Path:
    """Return the metadata file path for an upload directory."""
    return upload_dir / METADATA_FILENAME


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Load document metadata from JSON."""
    if not metadata_path.exists():
        return []

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Document metadata JSON is invalid: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Belge metadata kaydi okunamadi.",
        ) from exc
    except OSError as exc:
        logger.exception("Document metadata could not be read: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Belge metadata kaydi okunamadi.",
        ) from exc

    if not isinstance(payload, list):
        logger.error("Document metadata JSON has an invalid structure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Belge metadata kaydi gecersiz.",
        )
    return payload


def write_metadata(metadata_path: Path, records: list[dict[str, Any]]) -> None:
    """Write document metadata with an atomic replace."""
    temp_path = metadata_path.with_suffix(".json.tmp")
    try:
        temp_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(metadata_path)
    except OSError as exc:
        logger.exception("Document metadata could not be written: %s", exc)
        temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Belge metadata kaydi yazilamadi.",
        ) from exc


def find_duplicate_record(
    records: list[dict[str, Any]],
    sha256_hash: str,
) -> dict[str, Any] | None:
    """Find an existing metadata record by SHA-256 hash."""
    for record in records:
        if record.get("sha256") == sha256_hash:
            return record
    return None


def build_response(record: dict[str, Any], status_value: str, is_duplicate: bool) -> DocumentUploadResponse:
    """Build an upload response from a metadata record."""
    return DocumentUploadResponse(
        document_id=str(record["document_id"]),
        original_filename=str(record["original_filename"]),
        stored_filename=str(record["stored_filename"]),
        content_type=str(record["content_type"]),
        size_bytes=int(record["size_bytes"]),
        sha256=str(record["sha256"]),
        is_duplicate=is_duplicate,
        status=status_value,
        created_at=datetime.fromisoformat(str(record["created_at"])),
    )


async def save_upload_file(file: UploadFile) -> DocumentUploadResponse:
    """Validate and store an uploaded document under the configured upload directory."""
    safe_original_filename = sanitize_filename(file.filename)
    extension = validate_extension(safe_original_filename)
    content_type = validate_content_type(file.content_type)
    document_id = str(uuid4())
    stored_filename = f"{document_id}_{Path(safe_original_filename).stem}{extension}"
    upload_dir = Path(settings.upload_dir)
    metadata_path = get_metadata_path(upload_dir)
    destination = upload_dir / stored_filename
    temp_upload_path = upload_dir / f".{document_id}.uploading"
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        size_bytes = 0
        sha256 = hashlib.sha256()
        with temp_upload_path.open("xb") as output_file:
            while chunk := await file.read(CHUNK_SIZE_BYTES):
                size_bytes += len(chunk)
                if size_bytes > max_size_bytes:
                    output_file.close()
                    temp_upload_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Dosya boyutu izin verilen limiti asiyor.",
                    )
                sha256.update(chunk)
                output_file.write(chunk)

        if size_bytes == 0:
            temp_upload_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bos dosya yuklenemez.",
            )

        sha256_hash = sha256.hexdigest()
        with _METADATA_LOCK:
            records = load_metadata(metadata_path)
            duplicate_record = find_duplicate_record(records, sha256_hash)
            if duplicate_record is not None:
                temp_upload_path.unlink(missing_ok=True)
                logger.info(
                    "Duplicate document upload detected: document_id=%s",
                    duplicate_record.get("document_id"),
                )
                return build_response(
                    duplicate_record,
                    status_value="already_uploaded",
                    is_duplicate=True,
                )

            temp_upload_path.replace(destination)
            created_at = datetime.now(timezone.utc)
            record = {
                "document_id": document_id,
                "original_filename": safe_original_filename,
                "stored_filename": stored_filename,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "sha256": sha256_hash,
                "created_at": created_at.isoformat(),
            }
            records.append(record)
            write_metadata(metadata_path, records)

        logger.info("Document uploaded: document_id=%s size_bytes=%s", document_id, size_bytes)
        return build_response(record, status_value="uploaded", is_duplicate=False)
    except HTTPException:
        temp_upload_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        logger.exception("Document upload failed while saving file: %s", exc)
        temp_upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dosya kaydedilirken bir hata olustu.",
        ) from exc
    finally:
        await file.close()
