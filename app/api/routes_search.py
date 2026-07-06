"""Semantic document chunk search routes."""

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import SearchRequest, SearchResponse, SearchResultResponse
from app.rag.embedding_service import EmbeddingServiceError
from app.rag.retriever import Retriever
from app.rag.vector_store import (
    InvalidSearchQueryError,
    InvalidTopKError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/search", tags=["search"])
retriever = Retriever()


@router.post("", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Return document chunks relevant to a semantic search query."""
    try:
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
        response_results = [SearchResultResponse.model_validate(result) for result in results]
        return SearchResponse(
            query=request.query,
            result_count=len(response_results),
            results=response_results,
        )
    except (InvalidSearchQueryError, InvalidTopKError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except EmbeddingServiceError as exc:
        logger.warning("Semantic search embedding service is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding servisi kullanilamiyor.",
        ) from exc
    except VectorStoreError as exc:
        logger.error("Semantic search vector store operation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Arama islemi tamamlanamadi.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected semantic search error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Arama islemi tamamlanamadi.",
        ) from exc
