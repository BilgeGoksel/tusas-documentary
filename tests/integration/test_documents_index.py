"""Integration tests for processed-document indexing endpoints."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.rag.embedding_service import OllamaUnavailableError
from app.rag.models import Chunk
from app.rag.vector_store import VectorStoreError
from app.services import indexing_service

client = TestClient(app)
TEST_INDEX_ROOT = Path("tests/tmp_indexing")


@pytest.fixture
def tmp_path() -> Path:
    """Create a workspace-local temporary directory for indexing tests."""
    path = TEST_INDEX_ROOT / str(uuid4())
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def indexing_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    """Point upload, processed and indexed settings to isolated directories."""
    upload_dir = tmp_path / "uploads"
    processed_dir = tmp_path / "processed"
    indexed_dir = tmp_path / "indexed"
    upload_dir.mkdir()
    processed_dir.mkdir()
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "processed_dir", str(processed_dir))
    monkeypatch.setattr(settings, "indexed_dir", str(indexed_dir))
    return upload_dir, processed_dir, indexed_dir


def write_document_metadata(
    upload_dir: Path,
    document_id: str,
    original_filename: str,
    content_type: str,
) -> None:
    """Write one uploaded-document metadata record."""
    payload = [
        {
            "document_id": document_id,
            "original_filename": original_filename,
            "stored_filename": f"{document_id}_{original_filename}",
            "content_type": content_type,
        }
    ]
    (upload_dir / "documents.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def write_processed_result(
    processed_dir: Path,
    document_id: str,
    original_filename: str,
    texts: list[str],
    extraction_method: str,
) -> None:
    """Write a processing response compatible with the production cache."""
    pages = [
        {
            "page_number": index,
            "text": text,
            "extraction_method": extraction_method,
            "character_count": len(text),
            "requires_ocr": False,
            "confidence": 0.95 if extraction_method == "ocr" else None,
            "warnings": [],
        }
        for index, text in enumerate(texts, start=1)
    ]
    payload = {
        "document_id": document_id,
        "original_filename": original_filename,
        "page_count": len(pages),
        "total_character_count": sum(len(text) for text in texts),
        "processing_status": "processed",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "from_cache": False,
        "pages": pages,
    }
    (processed_dir / f"{document_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def prepare_document(
    dirs: tuple[Path, Path, Path],
    document_id: str = "doc-1",
    original_filename: str = "report.pdf",
    texts: list[str] | None = None,
    extraction_method: str = "native_pdf",
    content_type: str = "application/pdf",
) -> None:
    """Create upload and processing metadata for one document."""
    upload_dir, processed_dir, _ = dirs
    write_document_metadata(
        upload_dir, document_id, original_filename, content_type
    )
    write_processed_result(
        processed_dir,
        document_id,
        original_filename,
        texts if texts is not None else ["İşlenmiş belge metni."],
        extraction_method,
    )


def test_index_processed_pdf_preserves_chunk_sources(
    indexing_dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_document(indexing_dirs, texts=["Birinci sayfa", "İkinci sayfa"])
    captured: list[Chunk] = []
    monkeypatch.setattr(indexing_service, "add_chunks", lambda chunks: captured.extend(chunks))

    response = client.post("/api/v1/documents/doc-1/index")

    assert response.status_code == 200
    assert response.json()["page_count"] == 2
    assert {chunk.page_number for chunk in captured} == {1, 2}
    assert all(chunk.original_filename == "report.pdf" for chunk in captured)
    assert all(chunk.extraction_method == "native_pdf" for chunk in captured)


def test_index_processed_png_preserves_ocr_metadata(
    indexing_dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_document(
        indexing_dirs,
        original_filename="scan.png",
        extraction_method="ocr",
        content_type="image/png",
    )
    captured: list[Chunk] = []
    monkeypatch.setattr(indexing_service, "add_chunks", lambda chunks: captured.extend(chunks))

    response = client.post("/api/v1/documents/doc-1/index")

    assert response.status_code == 200
    assert captured[0].original_filename == "scan.png"
    assert captured[0].extraction_method == "ocr"


def test_unknown_document_returns_404(indexing_dirs: tuple[Path, Path, Path]) -> None:
    response = client.post("/api/v1/documents/missing/index")

    assert response.status_code == 404
    assert response.json()["detail"] == "Belge bulunamadi."


def test_unprocessed_document_returns_409(
    indexing_dirs: tuple[Path, Path, Path]
) -> None:
    upload_dir, _, _ = indexing_dirs
    write_document_metadata(upload_dir, "doc-1", "report.pdf", "application/pdf")

    response = client.post("/api/v1/documents/doc-1/index")

    assert response.status_code == 409


def test_empty_processed_document_returns_422(
    indexing_dirs: tuple[Path, Path, Path]
) -> None:
    prepare_document(indexing_dirs, texts=["  ", "\n\t"])

    response = client.post("/api/v1/documents/doc-1/index")

    assert response.status_code == 422


def test_first_index_is_not_cached_and_summary_contains_no_text(
    indexing_dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_document(indexing_dirs)
    monkeypatch.setattr(indexing_service, "add_chunks", lambda chunks: None)

    response = client.post("/api/v1/documents/doc-1/index")

    assert response.status_code == 200
    assert response.json()["from_cache"] is False
    summary = json.loads((indexing_dirs[2] / "doc-1.json").read_text(encoding="utf-8"))
    assert "text" not in json.dumps(summary)
    assert "path" not in summary


def test_second_index_uses_cached_summary(
    indexing_dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_document(indexing_dirs)
    calls = 0

    def fake_add(chunks: list[Chunk]) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(indexing_service, "add_chunks", fake_add)

    first = client.post("/api/v1/documents/doc-1/index")
    second = client.post("/api/v1/documents/doc-1/index")

    assert first.json()["from_cache"] is False
    assert second.json()["from_cache"] is True
    assert calls == 1


def test_force_reindexes_document(
    indexing_dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_document(indexing_dirs)
    calls = 0

    def fake_add(chunks: list[Chunk]) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(indexing_service, "add_chunks", fake_add)
    client.post("/api/v1/documents/doc-1/index")

    response = client.post("/api/v1/documents/doc-1/index?force=true")

    assert response.status_code == 200
    assert response.json()["from_cache"] is False
    assert calls == 2


def test_force_reindex_keeps_other_document_chunks(
    indexing_dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_document(indexing_dirs)
    stored = {"doc-other": ["other-chunk"], "doc-1": ["old-chunk"]}

    def fake_add(chunks: list[Chunk]) -> None:
        document_id = chunks[0].document_id
        stored[document_id] = [chunk.chunk_id for chunk in chunks]

    monkeypatch.setattr(indexing_service, "add_chunks", fake_add)

    response = client.post("/api/v1/documents/doc-1/index?force=true")

    assert response.status_code == 200
    assert stored["doc-other"] == ["other-chunk"]
    assert stored["doc-1"] != ["old-chunk"]


def test_embedding_failure_returns_503(
    indexing_dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_document(indexing_dirs)
    monkeypatch.setattr(
        indexing_service,
        "add_chunks",
        lambda chunks: (_ for _ in ()).throw(OllamaUnavailableError("kapali")),
    )

    response = client.post("/api/v1/documents/doc-1/index")

    assert response.status_code == 503
    assert "path" not in response.text.lower()


def test_vector_store_failure_returns_500(
    indexing_dirs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_document(indexing_dirs)
    monkeypatch.setattr(
        indexing_service,
        "add_chunks",
        lambda chunks: (_ for _ in ()).throw(VectorStoreError("db error")),
    )

    response = client.post("/api/v1/documents/doc-1/index")

    assert response.status_code == 500
    assert response.json()["detail"] == "Belge vector store'a kaydedilemedi."
