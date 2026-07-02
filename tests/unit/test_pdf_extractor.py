"""Tests for native PDF page text extraction."""

import shutil
from pathlib import Path
from uuid import uuid4

import fitz
import pytest

from app.core.config import settings
from app.document_processing.pdf_extractor import (
    EncryptedPDFError,
    InvalidPDFError,
    extract_pdf_pages,
)

TEST_PDF_ROOT = Path("tests/generated_pdfs")
ARIAL_FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")


@pytest.fixture
def pdf_tmp_path() -> Path:
    """Create a workspace-local temporary path for generated test PDFs."""
    path = TEST_PDF_ROOT / str(uuid4())
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def create_pdf(path: Path, page_texts: list[str], use_arial: bool = False) -> None:
    """Create a small test PDF with one page per text entry."""
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if not text:
            continue
        if use_arial:
            page.insert_font(fontname="Arial", fontfile=str(ARIAL_FONT_PATH))
            page.insert_text((72, 72), text, fontname="Arial", fontsize=12)
        else:
            page.insert_text((72, 72), text, fontsize=12)
    document.save(path)
    document.close()


def test_extract_single_page_text_pdf(pdf_tmp_path: Path) -> None:
    """Extract text from a single-page PDF."""
    pdf_path = pdf_tmp_path / "single.pdf"
    create_pdf(pdf_path, ["Hello PDF"])

    pages = extract_pdf_pages(pdf_path, document_id="doc-1", filename="single.pdf")

    assert len(pages) == 1
    assert pages[0].document_id == "doc-1"
    assert pages[0].filename == "single.pdf"
    assert pages[0].page_number == 1
    assert pages[0].text == "Hello PDF"
    assert pages[0].extraction_method == "native_pdf"
    assert pages[0].character_count == len("Hello PDF")
    assert pages[0].requires_ocr is True


def test_extract_multi_page_pdf(pdf_tmp_path: Path) -> None:
    """Extract text from each page of a multi-page PDF."""
    pdf_path = pdf_tmp_path / "multi.pdf"
    create_pdf(pdf_path, ["First page", "Second page", "Third page"])

    pages = extract_pdf_pages(pdf_path, document_id="doc-2", filename="multi.pdf")

    assert [page.text for page in pages] == ["First page", "Second page", "Third page"]
    assert [page.page_number for page in pages] == [1, 2, 3]


def test_extract_empty_page_keeps_empty_result(pdf_tmp_path: Path) -> None:
    """Keep an empty page as an empty extracted page result."""
    pdf_path = pdf_tmp_path / "empty-page.pdf"
    create_pdf(pdf_path, [""])

    pages = extract_pdf_pages(pdf_path, document_id="doc-3", filename="empty-page.pdf")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == ""
    assert pages[0].character_count == 0
    assert pages[0].requires_ocr is True


def test_extract_corrupt_pdf_raises_controlled_error(pdf_tmp_path: Path) -> None:
    """Raise a controlled error for corrupt PDF files."""
    pdf_path = pdf_tmp_path / "corrupt.pdf"
    pdf_path.write_bytes(b"not a valid pdf")

    with pytest.raises(InvalidPDFError):
        extract_pdf_pages(pdf_path, document_id="doc-4", filename="corrupt.pdf")


def test_extract_encrypted_pdf_raises_controlled_error(pdf_tmp_path: Path) -> None:
    """Raise a controlled error for encrypted PDF files."""
    source_path = pdf_tmp_path / "source.pdf"
    encrypted_path = pdf_tmp_path / "encrypted.pdf"
    create_pdf(source_path, ["Secret text"])
    source_document = fitz.open(source_path)
    source_document.save(
        encrypted_path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-pass",
        user_pw="user-pass",
    )
    source_document.close()

    with pytest.raises(EncryptedPDFError):
        extract_pdf_pages(encrypted_path, document_id="doc-5", filename="encrypted.pdf")


def test_extract_preserves_turkish_characters(pdf_tmp_path: Path) -> None:
    """Preserve Turkish characters in extracted native PDF text."""
    pdf_path = pdf_tmp_path / "turkish.pdf"
    text = "İstanbul Türkçe ğüşöçı"
    create_pdf(pdf_path, [text], use_arial=True)

    pages = extract_pdf_pages(pdf_path, document_id="doc-6", filename="turkish.pdf")

    assert pages[0].text == text


def test_extract_page_numbers_start_at_one(pdf_tmp_path: Path) -> None:
    """Expose page numbers with one-based indexing."""
    pdf_path = pdf_tmp_path / "numbers.pdf"
    create_pdf(pdf_path, ["A", "B"])

    pages = extract_pdf_pages(pdf_path, document_id="doc-7", filename="numbers.pdf")

    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_extract_marks_short_text_page_as_requiring_ocr(pdf_tmp_path: Path) -> None:
    """Mark pages below the configured text threshold as requiring OCR."""
    pdf_path = pdf_tmp_path / "short-text.pdf"
    create_pdf(pdf_path, ["Short text"])

    pages = extract_pdf_pages(pdf_path, document_id="doc-8", filename="short-text.pdf")

    assert pages[0].character_count < settings.pdf_ocr_min_text_chars
    assert pages[0].requires_ocr is True


def test_extract_marks_long_text_page_as_not_requiring_ocr(pdf_tmp_path: Path) -> None:
    """Do not require OCR when native PDF text meets the configured threshold."""
    pdf_path = pdf_tmp_path / "long-text.pdf"
    text = "This page contains enough native PDF text for extraction."
    create_pdf(pdf_path, [text])

    pages = extract_pdf_pages(pdf_path, document_id="doc-9", filename="long-text.pdf")

    assert pages[0].character_count >= settings.pdf_ocr_min_text_chars
    assert pages[0].requires_ocr is False


def test_extract_uses_configured_ocr_threshold(
    monkeypatch: pytest.MonkeyPatch,
    pdf_tmp_path: Path,
) -> None:
    """Use the configured threshold when deciding whether OCR is needed."""
    pdf_path = pdf_tmp_path / "custom-threshold.pdf"
    create_pdf(pdf_path, ["Custom threshold text"])
    monkeypatch.setattr(settings, "pdf_ocr_min_text_chars", 5)

    pages = extract_pdf_pages(
        pdf_path,
        document_id="doc-10",
        filename="custom-threshold.pdf",
    )

    assert pages[0].character_count >= 5
    assert pages[0].requires_ocr is False
