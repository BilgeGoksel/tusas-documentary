"""Orchestrate retrieval, grounded prompting and local answer generation."""

import logging

from app.core.config import settings
from app.models.schemas import QAResponse, QASourceResponse
from app.rag.answer_generator import generate_answer
from app.rag.prompt_builder import build_grounded_prompt_result
from app.rag.retriever import RetrievalResult, Retriever
from app.rag.vector_store import InvalidSearchQueryError, InvalidTopKError

NOT_FOUND_ANSWER = "Bu bilgi yüklenen belgelerde bulunamadı."
SNIPPET_MAX_CHARACTERS = 200

logger = logging.getLogger(__name__)
retriever = Retriever()


def answer_question(
    query: str,
    document_ids: list[str] | None = None,
    top_k: int = 6,
) -> QAResponse:
    """Answer a question using only sufficiently relevant document chunks."""
    _validate_request(query, top_k)
    retrieved_chunks = retriever.retrieve(
        query=query,
        top_k=top_k,
        document_ids=document_ids,
    )
    retrieved_count = len(retrieved_chunks)
    prompt_chunks = _select_prompt_chunks(
        retrieved_chunks,
        minimum_score=settings.retrieval_min_score,
    )
    if not prompt_chunks:
        best_score = max(
            (chunk["similarity_score"] for chunk in retrieved_chunks),
            default=None,
        )
        rejection_reason = "no_results" if not retrieved_chunks else "below_threshold"
        logger.debug(
            "QA retrieval rejected: best_similarity_score=%s "
            "threshold_used=%s retrieved_chunk_count=%s rejection_reason=%s",
            best_score,
            settings.retrieval_min_score,
            retrieved_count,
            rejection_reason,
        )
        return _not_found_response(retrieved_count, top_k)

    prompt = build_grounded_prompt_result(query, prompt_chunks)
    answer = generate_answer(prompt.messages)
    sources = [
        QASourceResponse(
            source_number=reference.source_number,
            document_id=reference.document_id,
            original_filename=reference.original_filename,
            page_number=reference.page_number,
            chunk_id=reference.chunk_id,
            similarity_score=chunk["similarity_score"],
            snippet=_build_snippet(chunk["text"]),
        )
        for reference, chunk in zip(prompt.sources, prompt_chunks, strict=True)
    ]
    return QAResponse(
        answer=answer,
        found_in_documents=True,
        sources=sources,
        retrieved_chunk_count=retrieved_count,
        model=settings.ollama_chat_model,
        top_k=top_k,
    )


def _validate_request(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise InvalidSearchQueryError("Arama sorgusu bos olamaz.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise InvalidTopKError("top_k 1 ile 20 arasinda olmalidir.")


def _select_prompt_chunks(
    chunks: list[RetrievalResult],
    minimum_score: float,
) -> list[RetrievalResult]:
    selected: list[RetrievalResult] = []
    seen: set[tuple[str, int, str]] = set()
    for chunk in chunks:
        if chunk["similarity_score"] < minimum_score:
            continue
        identity = (chunk["document_id"], chunk["page_number"], chunk["chunk_id"])
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(chunk)
    return selected


def _build_snippet(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= SNIPPET_MAX_CHARACTERS:
        return normalized
    return f"{normalized[: SNIPPET_MAX_CHARACTERS - 1].rstrip()}…"


def _not_found_response(retrieved_count: int, top_k: int) -> QAResponse:
    return QAResponse(
        answer=NOT_FOUND_ANSWER,
        found_in_documents=False,
        sources=[],
        retrieved_chunk_count=retrieved_count,
        model=settings.ollama_chat_model,
        top_k=top_k,
    )


__all__ = ["NOT_FOUND_ANSWER", "answer_question"]
