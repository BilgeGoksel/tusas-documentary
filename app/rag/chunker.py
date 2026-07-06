"""Page-aware text chunking that preserves document source metadata."""

from hashlib import sha256

from app.core.config import settings
from app.document_processing.models import PageContent
from app.rag.models import Chunk

MIN_BOUNDARY_RATIO = 0.6


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split extracted pages into deterministic, page-bounded chunks."""
    effective_size = settings.chunk_size if chunk_size is None else chunk_size
    effective_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    _validate_chunk_settings(effective_size, effective_overlap)

    chunks: list[Chunk] = []
    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        for chunk_index, text in enumerate(
            _split_text(page_text, effective_size, effective_overlap)
        ):
            chunks.append(
                Chunk(
                    chunk_id=_create_chunk_id(
                        page=page,
                        chunk_index=chunk_index,
                        text=text,
                        chunk_size=effective_size,
                        chunk_overlap=effective_overlap,
                    ),
                    document_id=page.document_id,
                    original_filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                    character_count=len(text),
                    extraction_method=page.extraction_method,
                )
            )
    return chunks


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size sifirdan buyuk olmalidir.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap negatif olamaz.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap, chunk_size degerinden kucuk olmalidir.")


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))
        end = _find_end_boundary(text, start, maximum_end, chunk_size)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break

        next_start = max(0, end - chunk_overlap)
        next_start = _move_to_word_start(text, next_start, end)
        if next_start <= start:
            next_start = end
        start = next_start
    return pieces


def _find_end_boundary(text: str, start: int, maximum_end: int, chunk_size: int) -> int:
    if maximum_end == len(text):
        return maximum_end

    minimum_end = start + int(chunk_size * MIN_BOUNDARY_RATIO)
    search_area = text[minimum_end:maximum_end]
    for separator in ("\n\n", "\n", " ", "\t"):
        position = search_area.rfind(separator)
        if position >= 0:
            return minimum_end + position + len(separator)
    return maximum_end


def _move_to_word_start(text: str, position: int, upper_bound: int) -> int:
    while position < upper_bound and not text[position].isspace():
        position += 1
    while position < upper_bound and text[position].isspace():
        position += 1
    return position


def _create_chunk_id(
    page: PageContent,
    chunk_index: int,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    identity = "\x1f".join(
        (
            page.document_id,
            page.filename,
            str(page.page_number),
            str(chunk_index),
            str(chunk_size),
            str(chunk_overlap),
            page.extraction_method,
            text,
        )
    )
    return sha256(identity.encode("utf-8")).hexdigest()
