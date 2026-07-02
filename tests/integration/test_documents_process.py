"""Document processing endpoint tests."""

import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from app.core.config import settings
from app.document_processing.models import ExtractedPage
from app.document_processing.ocr_service import OCRUnavailableError
from app.document_processing.pdf_extractor import InvalidPDFError
import app.services.document_processing_service as processing_service
from app.main import app

client = TestClient(app)
TEST_PROCESS_ROOT = Path("tests/tmp_process")


@pytest.fixture
def process_tmp_path() -> Path:
    """Create a workspace-local temporary path for processing endpoint tests."""
    path = TEST_PROCESS_ROOT / str(uuid4())
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def processing_dirs(monkeypatch: pytest.MonkeyPatch, process_tmp_path: Path) -> tuple[Path, Path]:
    """Configure upload and processed directories for a test."""
    upload_dir = process_tmp_path / "uploads"
    processed_dir = process_tmp_path / "processed"
    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "processed_dir", str(processed_dir))
    return upload_dir, processed_dir


def write_metadata(upload_dir: Path, record: dict[str, object]) -> None:
    """Write a single document metadata record."""
    (upload_dir / "documents.json").write_text(
        json.dumps([record], ensure_ascii=False),
        encoding="utf-8",
    )


def make_record(
    document_id: str,
    stored_filename: str,
    original_filename: str,
    content_type: str,
) -> dict[str, object]:
    """Build an upload metadata record for tests."""
    return {
        "document_id": document_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "content_type": content_type,
        "size_bytes": 10,
        "sha256": "abc",
        "created_at": "2026-07-02T00:00:00+00:00",
    }


def fake_page(
    document_id: str,
    filename: str,
    text: str,
    extraction_method: str,
    page_number: int = 1,
    confidence: float | None = None,
    warnings: list[str] | None = None,
) -> ExtractedPage:
    """Build a fake processed page."""
    return ExtractedPage(
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        text=text,
        extraction_method=extraction_method,
        character_count=len(text),
        requires_ocr=False,
        average_confidence=confidence,
        warnings=warnings or [],
    )


def test_process_uploaded_pdf(monkeypatch: pytest.MonkeyPatch, processing_dirs: tuple[Path, Path]) -> None:
    """Process an uploaded PDF and return page results."""
    upload_dir, _ = processing_dirs
    document_id = "doc-pdf"
    stored_filename = "doc-pdf_report.pdf"
    (upload_dir / stored_filename).write_bytes(b"%PDF test")
    write_metadata(
        upload_dir,
        make_record(document_id, stored_filename, "report.pdf", "application/pdf"),
    )

    def fake_process_document(*args: object, **kwargs: object) -> list[ExtractedPage]:
        return [fake_page(document_id, "report.pdf", "Native PDF text", "native_pdf")]

    monkeypatch.setattr(processing_service, "process_document", fake_process_document)

    response = client.post(f"/api/v1/documents/{document_id}/process")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == document_id
    assert payload["original_filename"] == "report.pdf"
    assert payload["page_count"] == 1
    assert payload["total_character_count"] == len("Native PDF text")
    assert payload["processing_status"] == "processed"
    assert payload["from_cache"] is False
    assert payload["pages"][0]["extraction_method"] == "native_pdf"


def test_process_uploaded_image(monkeypatch: pytest.MonkeyPatch, processing_dirs: tuple[Path, Path]) -> None:
    """Process an uploaded image and return OCR metadata."""
    upload_dir, _ = processing_dirs
    document_id = "doc-image"
    stored_filename = "doc-image_scan.png"
    (upload_dir / stored_filename).write_bytes(b"png")
    write_metadata(
        upload_dir,
        make_record(document_id, stored_filename, "scan.png", "image/png"),
    )

    def fake_process_document(*args: object, **kwargs: object) -> list[ExtractedPage]:
        return [
            fake_page(
                document_id,
                "scan.png",
                "OCR text",
                "ocr",
                confidence=0.88,
                warnings=["OCR confidence degeri dusuk."],
            )
        ]

    monkeypatch.setattr(processing_service, "process_document", fake_process_document)

    response = client.post(f"/api/v1/documents/{document_id}/process")

    assert response.status_code == 200
    page = response.json()["pages"][0]
    assert page["page_number"] == 1
    assert page["text"] == "OCR text"
    assert page["confidence"] == 0.88
    assert page["warnings"] == ["OCR confidence degeri dusuk."]


def test_process_unknown_document_id_returns_404(processing_dirs: tuple[Path, Path]) -> None:
    """Unknown document ids return 404."""
    upload_dir, _ = processing_dirs
    write_metadata(
        upload_dir,
        make_record("other-doc", "other.pdf", "other.pdf", "application/pdf"),
    )

    response = client.post("/api/v1/documents/missing-doc/process")

    assert response.status_code == 404
    assert response.json()["detail"] == "Belge bulunamadi."


def test_process_missing_stored_file_returns_410(processing_dirs: tuple[Path, Path]) -> None:
    """Metadata records with missing stored files return 410."""
    upload_dir, _ = processing_dirs
    document_id = "doc-missing-file"
    write_metadata(
        upload_dir,
        make_record(document_id, "missing.pdf", "missing.pdf", "application/pdf"),
    )

    response = client.post(f"/api/v1/documents/{document_id}/process")

    assert response.status_code == 410
    assert response.json()["detail"] == "Kayitli belge dosyasi diskte bulunamadi."


def test_process_returns_cached_result(
    monkeypatch: pytest.MonkeyPatch,
    processing_dirs: tuple[Path, Path],
) -> None:
    """Return cached processing results by default."""
    upload_dir, _ = processing_dirs
    document_id = "doc-cache"
    stored_filename = "doc-cache.pdf"
    (upload_dir / stored_filename).write_bytes(b"%PDF test")
    write_metadata(
        upload_dir,
        make_record(document_id, stored_filename, "cache.pdf", "application/pdf"),
    )
    calls = 0

    def fake_process_document(*args: object, **kwargs: object) -> list[ExtractedPage]:
        nonlocal calls
        calls += 1
        return [fake_page(document_id, "cache.pdf", f"Processed {calls}", "native_pdf")]

    monkeypatch.setattr(processing_service, "process_document", fake_process_document)

    first_response = client.post(f"/api/v1/documents/{document_id}/process")
    second_response = client.post(f"/api/v1/documents/{document_id}/process")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert calls == 1
    assert first_response.json()["from_cache"] is False
    assert second_response.json()["from_cache"] is True
    assert second_response.json()["pages"][0]["text"] == "Processed 1"


def test_process_force_reprocesses_cached_result(
    monkeypatch: pytest.MonkeyPatch,
    processing_dirs: tuple[Path, Path],
) -> None:
    """force=true bypasses the cached processing result."""
    upload_dir, _ = processing_dirs
    document_id = "doc-force"
    stored_filename = "doc-force.pdf"
    (upload_dir / stored_filename).write_bytes(b"%PDF test")
    write_metadata(
        upload_dir,
        make_record(document_id, stored_filename, "force.pdf", "application/pdf"),
    )
    calls = 0

    def fake_process_document(*args: object, **kwargs: object) -> list[ExtractedPage]:
        nonlocal calls
        calls += 1
        return [fake_page(document_id, "force.pdf", f"Processed {calls}", "native_pdf")]

    monkeypatch.setattr(processing_service, "process_document", fake_process_document)

    client.post(f"/api/v1/documents/{document_id}/process")
    forced_response = client.post(f"/api/v1/documents/{document_id}/process?force=true")

    assert forced_response.status_code == 200
    assert calls == 2
    assert forced_response.json()["from_cache"] is False
    assert forced_response.json()["pages"][0]["text"] == "Processed 2"


def test_process_unsupported_format_returns_415(
    processing_dirs: tuple[Path, Path],
) -> None:
    """Unsupported metadata content types return 415."""
    upload_dir, _ = processing_dirs
    document_id = "doc-unsupported"
    stored_filename = "doc.txt"
    (upload_dir / stored_filename).write_text("plain text", encoding="utf-8")
    write_metadata(
        upload_dir,
        make_record(document_id, stored_filename, "doc.txt", "text/plain"),
    )

    response = client.post(f"/api/v1/documents/{document_id}/process")

    assert response.status_code == 415
    assert response.json()["detail"] == "Desteklenmeyen belge formati."


def test_process_corrupt_document_returns_422(
    monkeypatch: pytest.MonkeyPatch,
    processing_dirs: tuple[Path, Path],
) -> None:
    """Corrupt documents return a controlled 422 response."""
    upload_dir, _ = processing_dirs
    document_id = "doc-corrupt"
    stored_filename = "corrupt.pdf"
    (upload_dir / stored_filename).write_bytes(b"broken")
    write_metadata(
        upload_dir,
        make_record(document_id, stored_filename, "corrupt.pdf", "application/pdf"),
    )

    def fake_process_document(*args: object, **kwargs: object) -> list[ExtractedPage]:
        raise InvalidPDFError("PDF dosyasi okunamadi veya bozuk.")

    monkeypatch.setattr(processing_service, "process_document", fake_process_document)

    response = client.post(f"/api/v1/documents/{document_id}/process")

    assert response.status_code == 422
    assert response.json()["detail"] == "PDF dosyasi okunamadi veya bozuk."


def test_process_ocr_failure_returns_500(
    monkeypatch: pytest.MonkeyPatch,
    processing_dirs: tuple[Path, Path],
) -> None:
    """OCR failures return a controlled 500 response."""
    upload_dir, _ = processing_dirs
    document_id = "doc-ocr-failure"
    stored_filename = "scan.png"
    (upload_dir / stored_filename).write_bytes(b"png")
    write_metadata(
        upload_dir,
        make_record(document_id, stored_filename, "scan.png", "image/png"),
    )

    def fake_process_document(*args: object, **kwargs: object) -> list[ExtractedPage]:
        raise OCRUnavailableError("OCR servisi calistirilamadi.")

    monkeypatch.setattr(processing_service, "process_document", fake_process_document)

    response = client.post(f"/api/v1/documents/{document_id}/process")

    assert response.status_code == 500
    assert response.json()["detail"] == "Belge islenirken bir hata olustu."


def test_process_service_failure_returns_500(
    monkeypatch: pytest.MonkeyPatch,
    processing_dirs: tuple[Path, Path],
) -> None:
    """Unexpected processing service failures return a controlled 500 response."""
    upload_dir, _ = processing_dirs
    document_id = "doc-service-failure"
    stored_filename = "service.pdf"
    (upload_dir / stored_filename).write_bytes(b"%PDF test")
    write_metadata(
        upload_dir,
        make_record(document_id, stored_filename, "service.pdf", "application/pdf"),
    )

    def fake_process_document(*args: object, **kwargs: object) -> list[ExtractedPage]:
        raise RuntimeError("boom")

    monkeypatch.setattr(processing_service, "process_document", fake_process_document)

    response = client.post(f"/api/v1/documents/{document_id}/process")

    assert response.status_code == 500
    assert response.json()["detail"] == "Belge islenirken bir hata olustu."
