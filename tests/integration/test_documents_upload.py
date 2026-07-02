"""Document upload endpoint tests."""

import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from app.core.config import settings
from app.main import app

client = TestClient(app)
METADATA_FILENAME = "documents.json"
TEST_UPLOAD_ROOT = Path("tests/tmp_uploads")


@pytest.fixture
def tmp_path() -> Path:
    """Create a workspace-local temporary path for upload tests."""
    path = TEST_UPLOAD_ROOT / str(uuid4())
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    """Create a temporary upload directory."""
    return tmp_path / "uploads"


def configure_upload_settings(monkeypatch, upload_dir: Path, max_upload_size_mb: int = 1) -> None:
    """Point upload settings to a temporary test directory."""
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "max_upload_size_mb", max_upload_size_mb)


def upload_file(filename: str, content: bytes, content_type: str):
    """Post a single test file to the upload endpoint."""
    return client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, content, content_type)},
    )


def physical_document_files(upload_dir: Path) -> list[Path]:
    """Return stored document files excluding metadata."""
    return [
        path
        for path in upload_dir.iterdir()
        if path.is_file() and path.name != METADATA_FILENAME
    ]


def test_upload_valid_pdf(monkeypatch, upload_dir: Path) -> None:
    """Valid PDF uploads are stored and reported."""
    configure_upload_settings(monkeypatch, upload_dir)

    response = upload_file("report.pdf", b"%PDF-1.4\ncontent", "application/pdf")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "uploaded"
    assert payload["original_filename"] == "report.pdf"
    assert payload["content_type"] == "application/pdf"
    assert payload["size_bytes"] == len(b"%PDF-1.4\ncontent")
    assert payload["sha256"] == hashlib.sha256(b"%PDF-1.4\ncontent").hexdigest()
    assert payload["is_duplicate"] is False
    assert (upload_dir / payload["stored_filename"]).exists()
    assert (upload_dir / METADATA_FILENAME).exists()


def test_upload_valid_png(monkeypatch, upload_dir: Path) -> None:
    """Valid PNG uploads are stored and reported."""
    configure_upload_settings(monkeypatch, upload_dir)

    response = upload_file("image.png", b"\x89PNG\r\n\x1a\ncontent", "image/png")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "uploaded"
    assert payload["content_type"] == "image/png"
    assert payload["stored_filename"].endswith("_image.png")


def test_upload_rejects_unsupported_extension(monkeypatch, upload_dir: Path) -> None:
    """Unsupported file extensions are rejected."""
    configure_upload_settings(monkeypatch, upload_dir)

    response = upload_file("notes.txt", b"hello", "text/plain")

    assert response.status_code == 415
    assert not upload_dir.exists()


def test_upload_rejects_empty_file(monkeypatch, upload_dir: Path) -> None:
    """Empty files are rejected."""
    configure_upload_settings(monkeypatch, upload_dir)

    response = upload_file("empty.pdf", b"", "application/pdf")

    assert response.status_code == 400
    assert not list(upload_dir.iterdir())


def test_upload_rejects_oversized_file(monkeypatch, upload_dir: Path) -> None:
    """Files larger than the configured limit are rejected."""
    configure_upload_settings(monkeypatch, upload_dir, max_upload_size_mb=1)
    oversized_content = b"x" * ((1024 * 1024) + 1)

    response = upload_file("large.pdf", oversized_content, "application/pdf")

    assert response.status_code == 413
    assert not list(upload_dir.iterdir())


def test_upload_same_filename_does_not_collide(monkeypatch, upload_dir: Path) -> None:
    """Two uploads with the same original filename get unique stored names."""
    configure_upload_settings(monkeypatch, upload_dir)

    first_response = upload_file("same.pdf", b"first", "application/pdf")
    second_response = upload_file("same.pdf", b"second", "application/pdf")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_payload = first_response.json()
    second_payload = second_response.json()
    assert first_payload["document_id"] != second_payload["document_id"]
    assert first_payload["stored_filename"] != second_payload["stored_filename"]
    assert (upload_dir / first_payload["stored_filename"]).read_bytes() == b"first"
    assert (upload_dir / second_payload["stored_filename"]).read_bytes() == b"second"


def test_upload_same_file_twice_creates_one_physical_file(monkeypatch, upload_dir: Path) -> None:
    """Uploading identical content twice reuses the first stored document."""
    configure_upload_settings(monkeypatch, upload_dir)

    first_response = upload_file("duplicate.pdf", b"same-content", "application/pdf")
    second_response = upload_file("duplicate.pdf", b"same-content", "application/pdf")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_payload = first_response.json()
    second_payload = second_response.json()
    assert len(physical_document_files(upload_dir)) == 1
    assert second_payload["document_id"] == first_payload["document_id"]
    assert second_payload["is_duplicate"] is True
    assert second_payload["status"] == "already_uploaded"


def test_upload_same_name_different_content_creates_two_documents(
    monkeypatch,
    upload_dir: Path,
) -> None:
    """Same filename with different content is not treated as duplicate."""
    configure_upload_settings(monkeypatch, upload_dir)

    first_response = upload_file("same-name.pdf", b"content-one", "application/pdf")
    second_response = upload_file("same-name.pdf", b"content-two", "application/pdf")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_payload = first_response.json()
    second_payload = second_response.json()
    assert first_payload["document_id"] != second_payload["document_id"]
    assert second_payload["is_duplicate"] is False
    assert len(physical_document_files(upload_dir)) == 2


def test_upload_different_name_same_content_is_duplicate(
    monkeypatch,
    upload_dir: Path,
) -> None:
    """Different filenames with identical content are treated as duplicate."""
    configure_upload_settings(monkeypatch, upload_dir)

    first_response = upload_file("first.pdf", b"shared-content", "application/pdf")
    second_response = upload_file("second.pdf", b"shared-content", "application/pdf")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_payload = first_response.json()
    second_payload = second_response.json()
    assert second_payload["document_id"] == first_payload["document_id"]
    assert second_payload["original_filename"] == "first.pdf"
    assert second_payload["is_duplicate"] is True
    assert len(physical_document_files(upload_dir)) == 1


def test_upload_writes_metadata_json(monkeypatch, upload_dir: Path) -> None:
    """Successful uploads are recorded in metadata JSON."""
    configure_upload_settings(monkeypatch, upload_dir)
    content = b"metadata-content"

    response = upload_file("metadata.pdf", content, "application/pdf")

    assert response.status_code == 200
    payload = response.json()
    metadata = json.loads((upload_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert len(metadata) == 1
    metadata_record = metadata[0]
    assert metadata_record["document_id"] == payload["document_id"]
    assert metadata_record["original_filename"] == "metadata.pdf"
    assert metadata_record["stored_filename"] == payload["stored_filename"]
    assert metadata_record["content_type"] == "application/pdf"
    assert metadata_record["size_bytes"] == len(content)
    assert metadata_record["sha256"] == hashlib.sha256(content).hexdigest()
    assert isinstance(metadata_record["created_at"], str)


def test_upload_returns_controlled_error_for_corrupt_metadata(
    monkeypatch,
    upload_dir: Path,
) -> None:
    """Corrupt metadata JSON returns a controlled server error."""
    configure_upload_settings(monkeypatch, upload_dir)
    upload_dir.mkdir(parents=True)
    (upload_dir / METADATA_FILENAME).write_text("{broken-json", encoding="utf-8")

    response = upload_file("new.pdf", b"new-content", "application/pdf")

    assert response.status_code == 500
    assert response.json()["detail"] == "Belge metadata kaydi okunamadi."
    assert not physical_document_files(upload_dir)


def test_upload_sanitizes_path_traversal_filename(monkeypatch, upload_dir: Path) -> None:
    """Path traversal filenames are reduced to a safe basename."""
    configure_upload_settings(monkeypatch, upload_dir)

    response = upload_file("../secret.pdf", b"safe", "application/pdf")

    assert response.status_code == 200
    payload = response.json()
    assert payload["original_filename"] == "secret.pdf"
    assert "/" not in payload["stored_filename"]
    assert "\\" not in payload["stored_filename"]
    assert (upload_dir / payload["stored_filename"]).exists()
