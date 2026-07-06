"""Tests for vector-store-backed chunk retrieval."""

from unittest.mock import Mock

import pytest

from app.rag.embedding_service import OllamaUnavailableError
from app.rag.retriever import Retriever
from app.rag.vector_store import InvalidSearchQueryError, InvalidTopKError, SearchResult
from app.rag import retriever as retriever_module


def make_result(
    chunk_id: str,
    document_id: str,
    distance: float,
) -> SearchResult:
    """Build a vector search result with complete source metadata."""
    return SearchResult(
        chunk_id=chunk_id,
        text=f"{chunk_id} metni",
        document_id=document_id,
        original_filename=f"{document_id}.pdf",
        page_number=1,
        chunk_index=0,
        extraction_method="native_pdf",
        distance=distance,
    )


def test_retrieve_from_single_document(monkeypatch: pytest.MonkeyPatch) -> None:
    search = Mock(return_value=[make_result("chunk-1", "doc-1", 0.2)])
    monkeypatch.setattr(retriever_module, "search_similar", search)

    results = Retriever().retrieve("kanat açıklığı nedir?")

    assert results == [
        {
            "chunk_id": "chunk-1",
            "text": "chunk-1 metni",
            "document_id": "doc-1",
            "original_filename": "doc-1.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "extraction_method": "native_pdf",
            "similarity_score": 0.8,
        }
    ]


def test_retrieve_from_two_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retriever_module,
        "search_similar",
        Mock(
            return_value=[
                make_result("chunk-1", "doc-1", 0.1),
                make_result("chunk-2", "doc-2", 0.3),
            ]
        ),
    )

    results = Retriever().retrieve("soru")

    assert {result["document_id"] for result in results} == {"doc-1", "doc-2"}


def test_document_filter_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    search = Mock(return_value=[make_result("chunk-2", "doc-2", 0.1)])
    monkeypatch.setattr(retriever_module, "search_similar", search)

    results = Retriever().retrieve("soru", top_k=3, document_ids=["doc-2"])

    assert [result["document_id"] for result in results] == ["doc-2"]
    search.assert_called_once_with(
        query="soru", top_k=3, document_ids=["doc-2"]
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(query: str) -> None:
    with pytest.raises(InvalidSearchQueryError):
        Retriever().retrieve(query)


@pytest.mark.parametrize("top_k", [0, 21, True])
def test_top_k_boundary_is_enforced(top_k: int) -> None:
    with pytest.raises(InvalidTopKError):
        Retriever().retrieve("soru", top_k=top_k)


def test_results_are_sorted_by_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retriever_module,
        "search_similar",
        Mock(
            return_value=[
                make_result("least", "doc-1", 0.8),
                make_result("most", "doc-1", 0.1),
                make_result("middle", "doc-1", 0.4),
            ]
        ),
    )

    results = Retriever().retrieve("soru")

    assert [result["chunk_id"] for result in results] == ["most", "middle", "least"]
    assert [result["similarity_score"] for result in results] == pytest.approx(
        [0.9, 0.6, 0.2]
    )


def test_embedding_error_is_not_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retriever_module,
        "search_similar",
        Mock(side_effect=OllamaUnavailableError("Ollama kapalı")),
    )

    with pytest.raises(OllamaUnavailableError):
        Retriever().retrieve("soru")


def test_empty_collection_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever_module, "search_similar", Mock(return_value=[]))

    assert Retriever().retrieve("soru") == []
