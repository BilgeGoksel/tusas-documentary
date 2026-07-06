"""Persistent ChromaDB storage and retrieval for document chunks."""

from __future__ import annotations

import logging
from typing import TypedDict

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from app.rag.embedding_service import embed_text, embed_texts
from app.rag.models import Chunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when a ChromaDB operation cannot be completed."""


class InvalidSearchQueryError(ValueError):
    """Raised when a vector search query is empty."""


class InvalidTopKError(ValueError):
    """Raised when the requested result count is outside the supported range."""


class SearchResult(TypedDict):
    """A matching stored chunk and its source metadata."""

    chunk_id: str
    text: str
    document_id: str
    original_filename: str
    page_number: int
    chunk_index: int
    extraction_method: str
    distance: float


_client: chromadb.PersistentClient | None = None
_collection: Collection | None = None
_collection_key: tuple[str, str] | None = None


def add_chunks(chunks: list[Chunk]) -> None:
    """Embed and replace the indexed chunks for the supplied documents."""
    if not chunks:
        return

    embeddings = embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise VectorStoreError("Embedding sayisi chunk sayisiyla uyusmuyor.")

    chunks_by_document: dict[str, list[tuple[Chunk, list[float]]]] = {}
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunks_by_document.setdefault(chunk.document_id, []).append((chunk, embedding))

    collection = _get_collection()
    try:
        for document_id, entries in chunks_by_document.items():
            collection.delete(where={"document_id": document_id})
            collection.add(
                ids=[chunk.chunk_id for chunk, _ in entries],
                documents=[chunk.text for chunk, _ in entries],
                embeddings=[embedding for _, embedding in entries],
                metadatas=[_chunk_metadata(chunk) for chunk, _ in entries],
            )
    except Exception as exc:
        logger.exception("Document chunks could not be stored in ChromaDB.")
        raise VectorStoreError("Chunk kayitlari ChromaDB'ye yazilamadi.") from exc


def delete_document(document_id: str) -> None:
    """Delete every stored chunk belonging to one document."""
    try:
        _get_collection().delete(where={"document_id": document_id})
    except Exception as exc:
        logger.exception("Document chunks could not be deleted from ChromaDB.")
        raise VectorStoreError("Belge kayitlari ChromaDB'den silinemedi.") from exc


def document_exists(document_id: str) -> bool:
    """Return whether at least one chunk exists for a document."""
    try:
        result = _get_collection().get(
            where={"document_id": document_id},
            limit=1,
            include=[],
        )
        return bool(result["ids"])
    except Exception as exc:
        logger.exception("Document existence could not be checked in ChromaDB.")
        raise VectorStoreError("Belge kaydi ChromaDB'de kontrol edilemedi.") from exc


def search_similar(
    query: str,
    top_k: int = 5,
    document_ids: list[str] | None = None,
) -> list[SearchResult]:
    """Find the most similar chunks, optionally limited to selected documents."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidSearchQueryError("Arama sorgusu bos olamaz.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise InvalidTopKError("top_k 1 ile 20 arasinda olmalidir.")
    if document_ids is not None and not document_ids:
        return []

    collection = _get_collection()
    try:
        if collection.count() == 0:
            return []
    except Exception as exc:
        logger.exception("ChromaDB collection could not be inspected.")
        raise VectorStoreError("ChromaDB collection bilgisi okunamadi.") from exc

    query_embedding = embed_text(query)
    where = _document_filter(document_ids)
    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _parse_search_results(result)
    except Exception as exc:
        logger.exception("Similarity search failed in ChromaDB.")
        raise VectorStoreError("ChromaDB benzerlik aramasi basarisiz oldu.") from exc


def _get_collection() -> Collection:
    global _client, _collection, _collection_key

    key = (settings.chroma_persist_dir, settings.chroma_collection_name)
    if _collection is not None and _collection_key == key:
        return _collection

    try:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        _collection_key = key
        return _collection
    except Exception as exc:
        logger.exception("Persistent ChromaDB collection could not be opened.")
        raise VectorStoreError("ChromaDB collection acilamadi.") from exc


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    return {
        "document_id": chunk.document_id,
        "original_filename": chunk.original_filename,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "extraction_method": chunk.extraction_method,
    }


def _document_filter(document_ids: list[str] | None) -> dict[str, object] | None:
    if document_ids is None:
        return None
    unique_ids = list(dict.fromkeys(document_ids))
    if len(unique_ids) == 1:
        return {"document_id": unique_ids[0]}
    return {"document_id": {"$in": unique_ids}}


def _parse_search_results(result: dict[str, object]) -> list[SearchResult]:
    ids = result.get("ids") or [[]]
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    if not all(isinstance(values, list) and values for values in (ids, documents, metadatas, distances)):
        return []

    rows = zip(ids[0], documents[0], metadatas[0], distances[0], strict=True)
    parsed: list[SearchResult] = []
    try:
        for chunk_id, text, metadata, distance in rows:
            parsed.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    document_id=str(metadata["document_id"]),
                    original_filename=str(metadata["original_filename"]),
                    page_number=int(metadata["page_number"]),
                    chunk_index=int(metadata["chunk_index"]),
                    extraction_method=str(metadata["extraction_method"]),
                    distance=float(distance),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise VectorStoreError("ChromaDB arama sonucu gecersiz.") from exc
    return parsed


__all__ = [
    "InvalidSearchQueryError",
    "InvalidTopKError",
    "SearchResult",
    "VectorStoreError",
    "add_chunks",
    "delete_document",
    "document_exists",
    "search_similar",
]
