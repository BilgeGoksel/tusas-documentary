"""Tests for the hybrid document processor."""

import shutil
from pathlib import Path
from uuid import uuid4

import fitz
import pytest
from PIL import Image, ImageDraw

from app.document_processing.document_processor import (
    UnsupportedDocumentFormatError,
    process_document,
)
from app.document_processing.ocr_service import OCRUnavailableError

TEST_DOCUMENT_ROOT = Path("tests/generated_documents")


class FakeOCREngine:
    """Small fake OCR engine returning one line per OCR call."""

    def __init__(self, texts: list[str], should_fail: bool = False) -> None:
        self.texts = texts
        self.should_fail = should_fail
        self.calls = 0

    def predict(self, image: object) -> list[object]:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("OCR failed")
        text = self.texts[min(self.calls - 1, len(self.texts) - 1)]
        if not text:
            return []
        return [
            [
                [[20, 40], [300, 40], [300, 80], [20, 80]],
                (text, 0.93),
            ]
        ]


@pytest.fixture
def document_tmp_path() -> Path:
    """Create a workspace-local temporary path for generated documents."""
    path = TEST_DOCUMENT_ROOT / str(uuid4())
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def create_text_pdf(path: Path, page_texts: list[str]) -> None:
    """Create a text-based PDF with one page per text entry."""
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    document.save(path)
    document.close()


def create_text_image(path: Path, text: str, image_format: str = "PNG") -> None:
    """Create a simple text image."""
    image = Image.new("RGB", (420, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 60), text, fill="black")
    image.save(path, format=image_format)


def create_scanned_pdf(path: Path, image_texts: list[str]) -> None:
    """Create an image-only PDF with one scanned page per image text."""
    document = fitz.open()
    for text in image_texts:
        image_path = path.parent / f"{uuid4()}.png"
        create_text_image(image_path, text)
        page = document.new_page(width=420, height=160)
        page.insert_image(page.rect, filename=str(image_path))
        image_path.unlink(missing_ok=True)
    document.save(path)
    document.close()


def test_process_text_based_pdf_uses_native_text(document_tmp_path: Path) -> None:
    """Use native PDF text when pages do not require OCR."""
    pdf_path = document_tmp_path / "native.pdf"
    text = "This PDF page has enough native text to avoid OCR."
    create_text_pdf(pdf_path, [text])
    engine = FakeOCREngine(["should not be used"])

    pages = process_document(
        pdf_path,
        "doc-1",
        "native.pdf",
        "application/pdf",
        ocr_engine=engine,
    )

    assert len(pages) == 1
    assert pages[0].text == text
    assert pages[0].page_number == 1
    assert pages[0].extraction_method == "native_pdf"
    assert pages[0].requires_ocr is False
    assert engine.calls == 0


def test_process_scanned_pdf_applies_ocr_to_all_pages(document_tmp_path: Path) -> None:
    """OCR all pages in an image-only PDF."""
    pdf_path = document_tmp_path / "scanned.pdf"
    create_scanned_pdf(pdf_path, ["Scanned page"])
    engine = FakeOCREngine(["OCR scanned page"])

    pages = process_document(
        pdf_path,
        "doc-2",
        "scanned.pdf",
        "application/pdf",
        ocr_engine=engine,
    )

    assert len(pages) == 1
    assert pages[0].text == "OCR scanned page"
    assert pages[0].page_number == 1
    assert pages[0].extraction_method == "ocr_pdf"
    assert pages[0].requires_ocr is False
    assert engine.calls == 1


def test_process_hybrid_pdf_preserves_order_and_ocrs_only_needed_page(
    document_tmp_path: Path,
) -> None:
    """Use native text and OCR together while preserving page order."""
    pdf_path = document_tmp_path / "hybrid.pdf"
    native_text = "The first page contains enough searchable native PDF text."
    scanned_image_path = document_tmp_path / "scanned-page.png"
    create_text_image(scanned_image_path, "Scanned second page")

    document = fitz.open()
    first_page = document.new_page(width=420, height=160)
    first_page.insert_text((72, 72), native_text, fontsize=12)
    second_page = document.new_page(width=420, height=160)
    second_page.insert_image(second_page.rect, filename=str(scanned_image_path))
    document.save(pdf_path)
    document.close()

    engine = FakeOCREngine(["OCR second page"])

    pages = process_document(
        pdf_path,
        "doc-3",
        "hybrid.pdf",
        "application/pdf",
        ocr_engine=engine,
    )

    assert [page.page_number for page in pages] == [1, 2]
    assert [page.extraction_method for page in pages] == ["native_pdf", "ocr_pdf"]
    assert pages[0].text == native_text
    assert pages[1].text == "OCR second page"
    assert engine.calls == 1


def test_process_png_uses_image_ocr(document_tmp_path: Path) -> None:
    """Process PNG documents through image OCR."""
    image_path = document_tmp_path / "image.png"
    create_text_image(image_path, "PNG text")

    pages = process_document(
        image_path,
        "doc-4",
        "image.png",
        "image/png",
        ocr_engine=FakeOCREngine(["PNG OCR text"]),
    )

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == "PNG OCR text"
    assert pages[0].extraction_method == "ocr"


def test_process_jpg_uses_image_ocr(document_tmp_path: Path) -> None:
    """Process JPG documents through image OCR."""
    image_path = document_tmp_path / "image.jpg"
    create_text_image(image_path, "JPG text", image_format="JPEG")

    pages = process_document(
        image_path,
        "doc-5",
        "image.jpg",
        "image/jpeg",
        ocr_engine=FakeOCREngine(["JPG OCR text"]),
    )

    assert pages[0].text == "JPG OCR text"
    assert pages[0].extraction_method == "ocr"


def test_process_unsupported_format_raises_controlled_error(document_tmp_path: Path) -> None:
    """Reject unsupported document formats."""
    file_path = document_tmp_path / "notes.txt"
    file_path.write_text("plain text", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentFormatError):
        process_document(file_path, "doc-6", "notes.txt", "text/plain")


def test_process_pdf_ocr_failure_raises_controlled_error(document_tmp_path: Path) -> None:
    """Surface OCR backend failures as controlled errors."""
    pdf_path = document_tmp_path / "scanned-failure.pdf"
    create_scanned_pdf(pdf_path, ["Scanned page"])

    with pytest.raises(OCRUnavailableError):
        process_document(
            pdf_path,
            "doc-7",
            "scanned-failure.pdf",
            "application/pdf",
            ocr_engine=FakeOCREngine([""], should_fail=True),
        )
