"""Tests for frontend QA API helpers."""

from unittest.mock import Mock

import requests

from frontend import streamlit_app


def test_ask_qa_sends_selected_documents_and_top_k(monkeypatch) -> None:
    """Forward the question, selected document ids and retrieval count."""
    response = Mock()
    response.ok = True
    response.json.return_value = {
        "answer": "Yanıt [1]",
        "found_in_documents": True,
        "sources": [],
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(streamlit_app.requests, "post", post)

    payload, error = streamlit_app.ask_qa(
        "http://localhost:8000",
        query="Soru",
        document_ids=["doc-1", "doc-2"],
        top_k=4,
    )

    assert error is None
    assert payload == response.json.return_value
    post.assert_called_once_with(
        "http://localhost:8000/api/v1/qa",
        json={
            "query": "Soru",
            "document_ids": ["doc-1", "doc-2"],
            "top_k": 4,
        },
        timeout=streamlit_app.QA_REQUEST_TIMEOUT_SECONDS,
    )


def test_ask_qa_handles_backend_unavailable(monkeypatch) -> None:
    """Return a controlled UI error when the backend cannot be reached."""
    monkeypatch.setattr(
        streamlit_app.requests,
        "post",
        Mock(side_effect=requests.ConnectionError("private connection detail")),
    )

    payload, error = streamlit_app.ask_qa(
        "http://localhost:8000",
        query="Soru",
        document_ids=["doc-1"],
        top_k=5,
    )

    assert payload is None
    assert error == "Backend'e erisilemiyor."
    assert "private" not in error


def test_prepare_document_pipeline_runs_all_steps(monkeypatch) -> None:
    """Upload, process and prepare one document in the required order."""
    calls: list[str] = []
    file = Mock(name="file")
    file.name = "rapor.pdf"
    monkeypatch.setattr(
        streamlit_app,
        "upload_document",
        Mock(
            side_effect=lambda *args: (
                calls.append("upload")
                or {
                    "document_id": "doc-1",
                    "original_filename": "rapor.pdf",
                    "status": "uploaded",
                },
                None,
            )
        ),
    )
    monkeypatch.setattr(
        streamlit_app,
        "process_document",
        Mock(
            side_effect=lambda *args: (
                calls.append("process")
                or {
                    "processing_status": "processed",
                    "page_count": 3,
                    "from_cache": False,
                },
                None,
            )
        ),
    )
    monkeypatch.setattr(
        streamlit_app,
        "index_document",
        Mock(
            side_effect=lambda *args: (
                calls.append("index")
                or {
                    "indexing_status": "indexed",
                    "chunk_count": 7,
                    "indexed_at": "2026-07-06T10:00:00Z",
                    "from_cache": False,
                },
                None,
            )
        ),
    )
    steps: list[str] = []

    result = streamlit_app.prepare_document_pipeline(
        "http://localhost:8000",
        file,
        on_step=lambda stage, message, progress: steps.append(stage),
    )

    assert calls == ["upload", "process", "index"]
    assert steps == ["upload", "process", "index", "ready"]
    assert result["prepared"] is True
    assert result["document_id"] == "doc-1"
    assert result["page_count"] == 3
    assert result["chunk_count"] == 7


def test_prepare_document_pipeline_stops_failed_file_at_process(monkeypatch) -> None:
    """Preserve the failing stage and avoid later calls for that document."""
    file = Mock(name="file")
    file.name = "bozuk.png"
    monkeypatch.setattr(
        streamlit_app,
        "upload_document",
        Mock(return_value=({"document_id": "doc-bad"}, None)),
    )
    monkeypatch.setattr(
        streamlit_app,
        "process_document",
        Mock(return_value=(None, "OCR tamamlanamadi.")),
    )
    index = Mock()
    monkeypatch.setattr(streamlit_app, "index_document", index)

    result = streamlit_app.prepare_document_pipeline(
        "http://localhost:8000", file
    )

    assert result["prepared"] is False
    assert result["failed_stage"] == "process"
    assert result["error"] == "OCR tamamlanamadi."
    index.assert_not_called()
