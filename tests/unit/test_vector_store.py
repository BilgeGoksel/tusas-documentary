"""Tests for persistent ChromaDB document chunk storage."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from chromadb.api.client import SharedSystemClient

from app.core.config import settings
from app.rag.embedding_service import OllamaUnavailableError
from app.rag.models import Chunk
from app.rag import vector_store

TEST_CHROMA_ROOT = Path("tests/generated_chroma")

@pytest.fixture
def isolated_store(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Point ChromaDB at a disposable temporary directory."""
    path = TEST_CHROMA_ROOT / str(uuid4())
    monkeypatch.setattr(settings, "chroma_persist_dir", str(path))
    monkeypatch.setattr(settings, "chroma_collection_name", f"test_{uuid4().hex}")
    vector_store._client = None
    vector_store._collection = None
    vector_store._collection_key = None
    try:
        yield path
    finally:
        vector_store._collection = None
        vector_store._client = None
        vector_store._collection_key = None
        SharedSystemClient.clear_system_cache()


def make_chunk(
    chunk_id: str,
    document_id: str = "doc-1",
    text: str = "örnek belge metni",
    page_number: int = 1,
    chunk_index: int = 0,
) -> Chunk:
    """Build a chunk with complete source metadata."""
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        original_filename=f"{document_id}.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        character_count=len(text),
        extraction_method="native_pdf",
    )


def mock_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use deterministic local vectors instead of Ollama."""
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts: [[float(index + 1), 0.0, 1.0] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(vector_store, "embed_text", lambda text: [1.0, 0.0, 1.0])


def test_add_chunks_and_preserve_metadata(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_embeddings(monkeypatch)
    chunk = make_chunk("chunk-1", page_number=3, chunk_index=2)

    vector_store.add_chunks([chunk])

    collection = vector_store._get_collection()
    stored = collection.get(ids=["chunk-1"], include=["documents", "metadatas"])
    assert stored["ids"] == ["chunk-1"]
    assert stored["documents"] == [chunk.text]
    assert stored["metadatas"][0] == {
        "document_id": "doc-1",
        "original_filename": "doc-1.pdf",
        "page_number": 3,
        "chunk_index": 2,
        "extraction_method": "native_pdf",
    }


def test_readding_same_chunk_does_not_duplicate(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_embeddings(monkeypatch)
    chunk = make_chunk("chunk-1")

    vector_store.add_chunks([chunk])
    vector_store.add_chunks([chunk])

    assert vector_store._get_collection().count() == 1


def test_delete_document_removes_only_selected_document(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_embeddings(monkeypatch)
    vector_store.add_chunks(
        [make_chunk("chunk-1", "doc-1"), make_chunk("chunk-2", "doc-2")]
    )

    vector_store.delete_document("doc-1")

    assert vector_store.document_exists("doc-1") is False
    assert vector_store.document_exists("doc-2") is True
    assert vector_store._get_collection().count() == 1


def test_reindexing_one_document_keeps_other_documents(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_embeddings(monkeypatch)
    vector_store.add_chunks(
        [make_chunk("old-1", "doc-1"), make_chunk("other", "doc-2")]
    )

    vector_store.add_chunks([make_chunk("new-1", "doc-1", text="yenilendi")])

    ids = set(vector_store._get_collection().get(include=[])["ids"])
    assert ids == {"new-1", "other"}


def test_search_can_filter_by_document_id(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_embeddings(monkeypatch)
    vector_store.add_chunks(
        [make_chunk("chunk-1", "doc-1"), make_chunk("chunk-2", "doc-2")]
    )

    results = vector_store.search_similar("soru", document_ids=["doc-2"])

    assert [result["document_id"] for result in results] == ["doc-2"]
    assert results[0]["chunk_id"] == "chunk-2"
    assert isinstance(results[0]["distance"], float)


def test_top_k_limits_results(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_embeddings(monkeypatch)
    chunks = [make_chunk(f"chunk-{index}", text=f"metin {index}") for index in range(4)]
    vector_store.add_chunks(chunks)

    assert len(vector_store.search_similar("soru", top_k=2)) == 2


@pytest.mark.parametrize("top_k", [0, 21, True])
def test_invalid_top_k_is_rejected(
    isolated_store: Path, top_k: int
) -> None:
    with pytest.raises(vector_store.InvalidTopKError):
        vector_store.search_similar("soru", top_k=top_k)


def test_empty_collection_returns_empty_result(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embed = Mock(return_value=[1.0, 0.0])
    monkeypatch.setattr(vector_store, "embed_text", embed)

    assert vector_store.search_similar("soru") == []
    embed.assert_not_called()


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(isolated_store: Path, query: str) -> None:
    with pytest.raises(vector_store.InvalidSearchQueryError):
        vector_store.search_similar(query)


def test_embedding_error_is_not_wrapped(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_embeddings(monkeypatch)
    vector_store.add_chunks([make_chunk("chunk-1")])
    monkeypatch.setattr(
        vector_store,
        "embed_text",
        Mock(side_effect=OllamaUnavailableError("Ollama kapalı")),
    )

    with pytest.raises(OllamaUnavailableError):
        vector_store.search_similar("soru")
