"""Tests for the grounded question-answering endpoint."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api import routes_qa
from app.main import app
from app.models.schemas import QAResponse, QASourceResponse
from app.rag.answer_generator import ChatModelNotFoundError, OllamaUnavailableError
from app.rag.vector_store import VectorStoreError
from app.services.qa_service import NOT_FOUND_ANSWER

client = TestClient(app)


def found_response() -> QAResponse:
    """Build a complete successful QA response."""
    return QAResponse(
        answer="Belgeye dayalı cevap [1]",
        found_in_documents=True,
        sources=[
            QASourceResponse(
                source_number=1,
                document_id="doc-1",
                original_filename="belge.pdf",
                page_number=3,
                chunk_id="chunk-1",
                similarity_score=0.91,
                snippet="Kısa kaynak özeti.",
            )
        ],
        retrieved_chunk_count=1,
        model="qwen3:4b",
        top_k=6,
    )


def test_qa_valid_question(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = Mock(return_value=found_response())
    monkeypatch.setattr(routes_qa, "answer_question", answer)

    response = client.post("/api/v1/qa", json={"query": "Belgedeki bilgi nedir?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Belgeye dayalı cevap [1]"
    assert response.json()["found_in_documents"] is True
    answer.assert_called_once_with(
        query="Belgedeki bilgi nedir?", document_ids=None, top_k=6
    )


def test_qa_forwards_document_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = Mock(return_value=found_response())
    monkeypatch.setattr(routes_qa, "answer_question", answer)

    response = client.post(
        "/api/v1/qa",
        json={"query": "Soru", "document_ids": ["doc-1"], "top_k": 3},
    )

    assert response.status_code == 200
    answer.assert_called_once_with(
        query="Soru", document_ids=["doc-1"], top_k=3
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_qa_rejects_empty_query(query: str) -> None:
    response = client.post("/api/v1/qa", json={"query": query})

    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [0, 21, True])
def test_qa_rejects_invalid_top_k(top_k: object) -> None:
    response = client.post("/api/v1/qa", json={"query": "Soru", "top_k": top_k})

    assert response.status_code == 422


def test_qa_returns_200_when_answer_is_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    not_found = QAResponse(
        answer=NOT_FOUND_ANSWER,
        found_in_documents=False,
        sources=[],
        retrieved_chunk_count=2,
        model="qwen3:4b",
        top_k=6,
    )
    monkeypatch.setattr(routes_qa, "answer_question", Mock(return_value=not_found))

    response = client.post("/api/v1/qa", json={"query": "Bilinmeyen bilgi"})

    assert response.status_code == 200
    assert response.json()["answer"] == NOT_FOUND_ANSWER
    assert response.json()["found_in_documents"] is False
    assert response.json()["sources"] == []


def test_qa_returns_503_when_ollama_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes_qa,
        "answer_question",
        Mock(side_effect=OllamaUnavailableError("internal connection detail")),
    )

    response = client.post("/api/v1/qa", json={"query": "Soru"})

    assert response.status_code == 503
    assert "internal" not in response.text.lower()


def test_qa_returns_503_when_chat_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes_qa,
        "answer_question",
        Mock(side_effect=ChatModelNotFoundError("private model detail")),
    )

    response = client.post("/api/v1/qa", json={"query": "Soru"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Chat modeli kullanilamiyor."


def test_qa_handles_retriever_error_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes_qa,
        "answer_question",
        Mock(side_effect=VectorStoreError("C:/private/chroma/path")),
    )

    response = client.post("/api/v1/qa", json={"query": "Soru"})

    assert response.status_code == 500
    assert "private" not in response.text.lower()
    assert "path" not in response.text.lower()


def test_qa_serializes_all_source_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routes_qa, "answer_question", Mock(return_value=found_response())
    )

    response = client.post("/api/v1/qa", json={"query": "Soru"})

    assert response.status_code == 200
    assert response.json()["sources"] == [
        {
            "source_number": 1,
            "document_id": "doc-1",
            "original_filename": "belge.pdf",
            "page_number": 3,
            "chunk_id": "chunk-1",
            "similarity_score": 0.91,
            "snippet": "Kısa kaynak özeti.",
        }
    ]
