"""Manual OCR integration test that uses the real PaddleOCR engine when available."""

import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw

from app.document_processing.ocr_service import extract_image_text


pytestmark = pytest.mark.manual_ocr
TEST_MANUAL_OCR_ROOT = Path("tests/generated_manual_ocr")


def test_real_paddleocr_reads_simple_english_png() -> None:
    """Run a small real OCR smoke test when PaddleOCR is installed."""
    if os.getenv("RUN_MANUAL_OCR") != "1":
        pytest.skip("Set RUN_MANUAL_OCR=1 to run the real OCR integration test.")
    pytest.importorskip("paddleocr")
    test_dir = TEST_MANUAL_OCR_ROOT / str(uuid4())
    test_dir.mkdir(parents=True, exist_ok=False)
    image_path = test_dir / "manual-ocr.png"
    try:
        image = Image.new("RGB", (420, 140), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 50), "Manual OCR Test", fill="black")
        image.save(image_path)

        page = extract_image_text(image_path, "manual-doc", "manual-ocr.png")

        assert page.extraction_method == "ocr"
        assert page.page_number == 1
        assert page.requires_ocr is False
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
