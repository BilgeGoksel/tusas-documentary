"""Tests for end-to-end grounded question-answer orchestration."""

import logging
from unittest.mock import Mock

import pytest

from app.core.config import settings
from app.rag.answer_generator import (
    ChatModelNotFoundError,
    InvalidChatResponseError,
    OllamaUnavailableError,
)
from app.rag.retriever import RetrievalResult
from app.rag.vector_store import (
    InvalidSearchQueryError,
    InvalidTopKError,
    VectorStoreError,
)
from app.services import qa_service


def make_chunk(
    chunk_id: str = "chunk-1",
    score: float = 0.9,
    document_id: str = "doc-1",
    page_number: int = 2,
    text: str = "Belgede doğrulanmış cevap yer alır.",
    extraction_method: str = "native_pdf",
) -> RetrievalResult:
    """Build one retrieved chunk for QA tests."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        original_filename=f"{document_id}.pdf",
        page_number=page_number,
        chunk_index=0,
        extraction_method=extraction_method,
        similarity_score=score,
    )


def test_answer_found_in_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieve = Mock(return_value=[make_chunk()])
    monkeypatch.setattr(qa_service.retriever, "retrieve", retrieve)
    generate = Mock(return_value="Doğrulanmış cevap [1]")
    monkeypatch.setattr(qa_service, "generate_answer", generate)

    response = qa_service.answer_question("Cevap nedir?")

    assert response.answer == "Doğrulanmış cevap [1]"
    assert response.found_in_documents is True
    assert response.retrieved_chunk_count == 1
    assert response.model == settings.ollama_chat_model
    assert response.sources[0].source_number == 1
    generate.assert_called_once()
    retrieve.assert_called_once_with(query="Cevap nedir?", top_k=6, document_ids=None)
    assert response.top_k == 6


def test_question_not_found_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qa_service.retriever,
        "retrieve",
        Mock(return_value=[make_chunk(score=settings.retrieval_min_score - 0.01)]),
    )
    generate = Mock()
    monkeypatch.setattr(qa_service, "generate_answer", generate)

    response = qa_service.answer_question("Belgelerde olmayan soru")

    assert response.answer == qa_service.NOT_FOUND_ANSWER
    assert response.found_in_documents is False
    assert response.sources == []
    generate.assert_not_called()


def test_similarity_021_is_accepted_with_015_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "retrieval_min_score", 0.15)
    monkeypatch.setattr(
        qa_service.retriever,
        "retrieve",
        Mock(return_value=[make_chunk(score=0.21)]),
    )
    generate = Mock(return_value="OCR belgesine dayalı cevap [1]")
    monkeypatch.setattr(qa_service, "generate_answer", generate)

    response = qa_service.answer_question("Soru")

    assert response.found_in_documents is True
    assert response.sources[0].similarity_score == pytest.approx(0.21)
    generate.assert_called_once()


def test_similarity_010_is_rejected_with_safe_debug_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "retrieval_min_score", 0.15)
    private_text = "LOGA_GIRMEMESI_GEREKEN_BELGE_METNI"
    private_query = "LOGA_GIRMEMESI_GEREKEN_SORU"
    monkeypatch.setattr(
        qa_service.retriever,
        "retrieve",
        Mock(return_value=[make_chunk(score=0.10, text=private_text)]),
    )
    generate = Mock()
    monkeypatch.setattr(qa_service, "generate_answer", generate)

    with caplog.at_level(logging.DEBUG, logger=qa_service.__name__):
        response = qa_service.answer_question(private_query)

    assert response.found_in_documents is False
    assert "best_similarity_score=0.1" in caplog.text
    assert "threshold_used=0.15" in caplog.text
    assert "retrieved_chunk_count=1" in caplog.text
    assert "rejection_reason=below_threshold" in caplog.text
    assert private_query not in caplog.text
    assert private_text not in caplog.text
    generate.assert_not_called()


def test_ocr_like_relevant_chunk_with_low_score_calls_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "retrieval_min_score", 0.15)
    ocr_chunk = make_chunk(
        chunk_id="ocr-chunk",
        score=0.18,
        text="OCR ile çıkarılan ilgili kısa belge metni.",
        extraction_method="ocr",
    )
    monkeypatch.setattr(
        qa_service.retriever,
        "retrieve",
        Mock(return_value=[ocr_chunk]),
    )
    generate = Mock(return_value="OCR içeriğine dayalı cevap [1]")
    monkeypatch.setattr(qa_service, "generate_answer", generate)

    response = qa_service.answer_question("OCR belgesindeki bilgi nedir?")

    assert response.found_in_documents is True
    assert response.sources[0].chunk_id == "ocr-chunk"
    generate.assert_called_once()


def test_empty_retrieval_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_service.retriever, "retrieve", Mock(return_value=[]))
    generate = Mock()
    monkeypatch.setattr(qa_service, "generate_answer", generate)

    response = qa_service.answer_question("Soru")

    assert response.answer == qa_service.NOT_FOUND_ANSWER
    assert response.retrieved_chunk_count == 0
    generate.assert_not_called()


def test_only_chunks_meeting_threshold_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        make_chunk(
            "accepted", score=settings.retrieval_min_score, text="Kabul edilen içerik"
        ),
        make_chunk(
            "rejected",
            score=settings.retrieval_min_score - 0.001,
            text="Reddedilen içerik",
        ),
    ]
    monkeypatch.setattr(qa_service.retriever, "retrieve", Mock(return_value=chunks))
    generate = Mock(return_value="Yanıt [1]")
    monkeypatch.setattr(qa_service, "generate_answer", generate)

    response = qa_service.answer_question("Soru")

    assert response.retrieved_chunk_count == 2
    assert [source.chunk_id for source in response.sources] == ["accepted"]
    assert "Kabul edilen içerik" in generate.call_args.args[0][1]["content"]
    assert "Reddedilen içerik" not in generate.call_args.args[0][1]["content"]


def test_multiple_sources_and_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = make_chunk("chunk-1", page_number=2, text="İlk kaynak")
    duplicate = make_chunk("chunk-1", page_number=2, text="İlk kaynak")
    second = make_chunk("chunk-2", document_id="doc-2", page_number=5, text="İkinci kaynak")
    monkeypatch.setattr(
        qa_service.retriever, "retrieve", Mock(return_value=[first, duplicate, second])
    )
    monkeypatch.setattr(qa_service, "generate_answer", Mock(return_value="Yanıt [1] [2]"))

    response = qa_service.answer_question("Soru")

    assert [source.source_number for source in response.sources] == [1, 2]
    assert [source.chunk_id for source in response.sources] == ["chunk-1", "chunk-2"]
    assert response.sources[0].snippet == "İlk kaynak"


def test_document_filter_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieve = Mock(return_value=[make_chunk(document_id="doc-2")])
    monkeypatch.setattr(qa_service.retriever, "retrieve", retrieve)
    monkeypatch.setattr(qa_service, "generate_answer", Mock(return_value="Yanıt [1]"))

    qa_service.answer_question("Soru", document_ids=["doc-2"], top_k=3)

    retrieve.assert_called_once_with(
        query="Soru", top_k=3, document_ids=["doc-2"]
    )


def test_ollama_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_service.retriever, "retrieve", Mock(return_value=[make_chunk()]))
    monkeypatch.setattr(
        qa_service,
        "generate_answer",
        Mock(side_effect=OllamaUnavailableError("kapalı")),
    )

    with pytest.raises(OllamaUnavailableError):
        qa_service.answer_question("Soru")


def test_source_numbers_match_prompt_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [make_chunk("chunk-a"), make_chunk("chunk-b", page_number=4)]
    monkeypatch.setattr(qa_service.retriever, "retrieve", Mock(return_value=chunks))
    generate = Mock(return_value="Yanıt [1] [2]")
    monkeypatch.setattr(qa_service, "generate_answer", generate)

    response = qa_service.answer_question("Soru")
    context = generate.call_args.args[0][1]["content"]

    assert "[KAYNAK 1]" in context and "[KAYNAK 2]" in context
    assert [source.source_number for source in response.sources] == [1, 2]


def test_sources_return_even_when_llm_omits_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qa_service.retriever, "retrieve", Mock(return_value=[make_chunk()]))
    monkeypatch.setattr(
        qa_service, "generate_answer", Mock(return_value="Atıfsız ama grounded cevap")
    )

    response = qa_service.answer_question("Soru")

    assert "[1]" not in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].source_number == 1


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected_before_retrieval(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    retrieve = Mock()
    monkeypatch.setattr(qa_service.retriever, "retrieve", retrieve)

    with pytest.raises(InvalidSearchQueryError):
        qa_service.answer_question(query)

    retrieve.assert_not_called()


@pytest.mark.parametrize("top_k", [0, 21, True])
def test_invalid_top_k_is_rejected_before_retrieval(
    monkeypatch: pytest.MonkeyPatch, top_k: int
) -> None:
    retrieve = Mock()
    monkeypatch.setattr(qa_service.retriever, "retrieve", retrieve)

    with pytest.raises(InvalidTopKError):
        qa_service.answer_question("Soru", top_k=top_k)

    retrieve.assert_not_called()


def test_retriever_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qa_service.retriever,
        "retrieve",
        Mock(side_effect=VectorStoreError("internal vector detail")),
    )

    with pytest.raises(VectorStoreError):
        qa_service.answer_question("Soru")


@pytest.mark.parametrize(
    "error",
    [
        ChatModelNotFoundError("model bulunamadi"),
        InvalidChatResponseError("gecersiz cevap"),
    ],
)
def test_chat_errors_are_not_hidden(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(qa_service.retriever, "retrieve", Mock(return_value=[make_chunk()]))
    monkeypatch.setattr(qa_service, "generate_answer", Mock(side_effect=error))

    with pytest.raises(type(error)):
        qa_service.answer_question("Soru")
