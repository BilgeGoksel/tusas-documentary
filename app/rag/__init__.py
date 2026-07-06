"""Retrieval-augmented generation building blocks."""

from app.rag.chunker import chunk_pages
from app.rag.answer_generator import generate_answer
from app.rag.embedding_service import embed_text, embed_texts
from app.rag.models import Chunk
from app.rag.prompt_builder import (
    GroundedPromptResult,
    SourceReference,
    build_grounded_prompt,
    build_grounded_prompt_result,
)
from app.rag.retriever import Retriever, RetrievalResult
from app.rag.vector_store import (
    add_chunks,
    delete_document,
    document_exists,
    search_similar,
)

__all__ = [
    "Chunk",
    "GroundedPromptResult",
    "RetrievalResult",
    "Retriever",
    "SourceReference",
    "add_chunks",
    "generate_answer",
    "chunk_pages",
    "build_grounded_prompt",
    "build_grounded_prompt_result",
    "delete_document",
    "document_exists",
    "embed_text",
    "embed_texts",
    "search_similar",
]
