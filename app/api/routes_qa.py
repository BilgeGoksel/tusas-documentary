"""Grounded document question-answering routes."""

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import QARequest, QAResponse
from app.rag.answer_generator import (
    ChatGenerationError,
    ChatModelNotFoundError,
    OllamaUnavailableError,
)
from app.rag.embedding_service import EmbeddingServiceError
from app.rag.vector_store import (
    InvalidSearchQueryError,
    InvalidTopKError,
    VectorStoreError,
)
from app.services.qa_service import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


@router.post("", response_model=QAResponse)
def ask_question(request: QARequest) -> QAResponse:
    """Answer a question using only indexed document evidence."""
    try:
        return answer_question(
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )
    except (InvalidSearchQueryError, InvalidTopKError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ChatModelNotFoundError as exc:
        logger.warning("QA chat model is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat modeli kullanilamiyor.",
        ) from exc
    except OllamaUnavailableError as exc:
        logger.warning("QA Ollama service is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama servisi kullanilamiyor.",
        ) from exc
    except EmbeddingServiceError as exc:
        logger.warning("QA embedding service failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding servisi kullanilamiyor.",
        ) from exc
    except ChatGenerationError as exc:
        logger.error("QA chat generation failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Cevap uretilemedi.",
        ) from exc
    except VectorStoreError as exc:
        logger.error("QA retrieval operation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Belge aramasi tamamlanamadi.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected QA service error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Soru cevaplanirken bir hata olustu.",
        ) from exc
