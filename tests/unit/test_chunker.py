"""Tests for page-aware document chunking."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.document_processing.models import ExtractedPage
from app.rag.chunker import chunk_pages


def make_page(text: str, page_number: int = 1) -> ExtractedPage:
    """Build an extracted page for chunker tests."""
    return ExtractedPage(
        document_id="doc-1",
        filename="belge.pdf",
        page_number=page_number,
        text=text,
        extraction_method="native_pdf",
        character_count=len(text),
        requires_ocr=False,
    )


def test_single_short_page_is_not_lost() -> None:
    chunks = chunk_pages([make_page("Kısa ama anlamlı metin.")])

    assert len(chunks) == 1
    assert chunks[0].text == "Kısa ama anlamlı metin."
    assert chunks[0].character_count == len(chunks[0].text)


def test_long_page_creates_multiple_bounded_chunks() -> None:
    chunks = chunk_pages([make_page("kelime " * 500)], chunk_size=100, chunk_overlap=15)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_multiple_pages_keep_source_metadata_and_boundaries() -> None:
    chunks = chunk_pages(
        [make_page("Birinci sayfa " * 30, 1), make_page("İkinci sayfa " * 30, 2)],
        chunk_size=80,
        chunk_overlap=10,
    )

    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(
        ("Birinci" in chunk.text) != ("İkinci" in chunk.text) for chunk in chunks
    )
    assert [chunk.chunk_index for chunk in chunks if chunk.page_number == 2][0] == 0


def test_empty_page_does_not_create_chunk() -> None:
    assert chunk_pages([make_page(" \n\t ")]) == []


def test_turkish_characters_are_preserved() -> None:
    text = "Çığ ŞİİR öğütür; ıslak üzüm güzel."
    joined = " ".join(chunk.text for chunk in chunk_pages([make_page(text)]))

    assert joined == text


def test_chunks_contain_expected_overlap() -> None:
    chunks = chunk_pages(
        [make_page("bir iki üç dört beş altı yedi sekiz dokuz on")],
        chunk_size=25,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert any(word in chunks[1].text.split() for word in chunks[0].text.split()[-3:])


def test_chunk_ids_are_deterministic() -> None:
    page = make_page("Aynı belge ve ayarlar aynı kimliği üretir. " * 10)

    first = chunk_pages([page], chunk_size=80, chunk_overlap=10)
    second = chunk_pages([page], chunk_size=80, chunk_overlap=10)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_custom_chunk_size_is_applied() -> None:
    chunks = chunk_pages([make_page("uzunmetin" * 30)], chunk_size=40, chunk_overlap=5)

    assert all(chunk.character_count <= 40 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_chunk_arguments_raise_clear_error(
    chunk_size: int, chunk_overlap: int
) -> None:
    with pytest.raises(ValueError):
        chunk_pages([make_page("metin")], chunk_size, chunk_overlap)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100, _env_file=None)
