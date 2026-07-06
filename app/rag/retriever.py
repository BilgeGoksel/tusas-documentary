"""Retrieve the most relevant indexed chunks for a user query."""

from typing import TypedDict

from app.rag.vector_store import search_similar


class RetrievalResult(TypedDict):
    """A retrieved chunk with source metadata and relevance score."""

    chunk_id: str
    text: str
    document_id: str
    original_filename: str
    page_number: int
    chunk_index: int
    extraction_method: str
    similarity_score: float


class Retriever:
    """Retrieve and rank document chunks through the vector store."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Return relevant chunks ordered by descending similarity."""
        matches = search_similar(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )
        results = [
            RetrievalResult(
                chunk_id=match["chunk_id"],
                text=match["text"],
                document_id=match["document_id"],
                original_filename=match["original_filename"],
                page_number=match["page_number"],
                chunk_index=match["chunk_index"],
                extraction_method=match["extraction_method"],
                similarity_score=1.0 - match["distance"],
            )
            for match in matches
        ]
        return sorted(
            results,
            key=lambda result: result["similarity_score"],
            reverse=True,
        )


__all__ = ["RetrievalResult", "Retriever"]
