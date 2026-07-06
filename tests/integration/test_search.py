"""Tests for the semantic document search endpoint."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api import routes_search
from app.main import app
from app.rag.embedding_service import OllamaUnavailableError

client = TestClient(app)


def make_result(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
) -> dict[str, object]:
    """Build one complete mocked retrieval result."""
    return {
        "chunk_id": chunk_id,
        "text": "İlgili belge parçası",
        "document_id": document_id,
        "original_filename": f"{document_id}.pdf",
        "page_number": 2,
        "chunk_index": 1,
        "extraction_method": "native_pdf",
        "similarity_score": 0.91,
    }


def test_search_valid_query(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieve = Mock(return_value=[make_result()])
    monkeypatch.setattr(routes_search.retriever, "retrieve", retrieve)

    response = client.post("/api/v1/search", json={"query": "Kanat açıklığı nedir?"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "Kanat açıklığı nedir?",
        "result_count": 1,
        "results": [make_result()],
    }
    retrieve.assert_called_once_with(
        query="Kanat açıklığı nedir?", top_k=5, document_ids=None
    )


def test_search_forwards_document_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieve = Mock(return_value=[make_result(document_id="doc-2")])
    monkeypatch.setattr(routes_search.retriever, "retrieve", retrieve)

    response = client.post(
        "/api/v1/search",
        json={"query": "soru", "top_k": 3, "document_ids": ["doc-2"]},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["document_id"] == "doc-2"
    retrieve.assert_called_once_with(
        query="soru", top_k=3, document_ids=["doc-2"]
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_search_rejects_empty_query(query: str) -> None:
    response = client.post("/api/v1/search", json={"query": query})

    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [0, 21, True])
def test_search_rejects_invalid_top_k(top_k: int) -> None:
    response = client.post(
        "/api/v1/search", json={"query": "soru", "top_k": top_k}
    )

    assert response.status_code == 422


def test_search_returns_empty_list_when_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_search.retriever, "retrieve", Mock(return_value=[]))

    response = client.post("/api/v1/search", json={"query": "sonuçsuz soru"})

    assert response.status_code == 200
    assert response.json()["result_count"] == 0
    assert response.json()["results"] == []


def test_search_hides_unexpected_retriever_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes_search.retriever,
        "retrieve",
        Mock(side_effect=RuntimeError("C:/internal/private/path")),
    )

    response = client.post("/api/v1/search", json={"query": "gizli soru"})

    assert response.status_code == 500
    assert "internal" not in response.text.lower()
    assert "path" not in response.text.lower()


def test_search_returns_503_when_ollama_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes_search.retriever,
        "retrieve",
        Mock(side_effect=OllamaUnavailableError("connection refused")),
    )

    response = client.post("/api/v1/search", json={"query": "soru"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding servisi kullanilamiyor."
