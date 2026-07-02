"""Tests for image preprocessing and OCR service behavior."""

import builtins
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw, ImageOps

from app.document_processing.image_preprocessor import (
    ImageTooSmallError,
    InvalidImageError,
    preprocess_image_for_ocr,
)
from app.document_processing.ocr_service import (
    OCRUnavailableError,
    extract_image_text,
    get_ocr_engine,
)
import app.document_processing.ocr_service as ocr_service

TEST_IMAGE_ROOT = Path("tests/generated_images")


class FakeOCREngine:
    """Small fake PaddleOCR-compatible engine for unit tests."""

    def __init__(self, result: list[object] | None = None, should_fail: bool = False) -> None:
        self.result = result if result is not None else []
        self.should_fail = should_fail
        self.predict_calls = 0

    def predict(self, image: object) -> list[object]:
        self.predict_calls += 1
        if self.should_fail:
            raise RuntimeError("OCR backend failed")
        return self.result


@pytest.fixture
def image_tmp_path() -> Path:
    """Create a workspace-local temporary path for generated test images."""
    path = TEST_IMAGE_ROOT / str(uuid4())
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def create_text_image(path: Path, text: str, image_format: str = "PNG") -> None:
    """Create a simple image containing text for OCR service tests."""
    image = Image.new("RGB", (420, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 50), text, fill="black")
    image.save(path, format=image_format)


def test_extract_turkish_text_png_with_mocked_ocr(image_tmp_path: Path) -> None:
    """Return OCR output for a PNG containing Turkish text."""
    image_path = image_tmp_path / "turkish.png"
    create_text_image(image_path, "İstanbul Türkçe")
    engine = FakeOCREngine(
        [
            [
                [[20, 50], [220, 50], [220, 80], [20, 80]],
                ("İstanbul Türkçe", 0.94),
            ]
        ]
    )

    page = extract_image_text(image_path, "doc-1", "turkish.png", ocr_engine=engine)

    assert engine.predict_calls == 1
    assert page.page_number == 1
    assert page.text == "İstanbul Türkçe"
    assert page.extraction_method == "ocr"
    assert page.character_count == len("İstanbul Türkçe")
    assert page.requires_ocr is False
    assert page.average_confidence == pytest.approx(0.94)


def test_extract_uses_paddleocr3_predict_without_cls(image_tmp_path: Path) -> None:
    """Use PaddleOCR 3.x predict(image) without cls arguments."""
    image_path = image_tmp_path / "predict.png"
    create_text_image(image_path, "Predict text")

    class PredictOnlyEngine:
        def __init__(self) -> None:
            self.received_image = None

        def predict(self, image: object) -> list[object]:
            self.received_image = image
            return [
                {
                    "rec_texts": ["Predict text"],
                    "rec_scores": [0.96],
                    "rec_polys": [
                        [[20, 50], [200, 50], [200, 80], [20, 80]],
                    ],
                }
            ]

    engine = PredictOnlyEngine()

    page = extract_image_text(image_path, "doc-predict", "predict.png", ocr_engine=engine)

    assert engine.received_image is not None
    assert page.text == "Predict text"
    assert page.average_confidence == pytest.approx(0.96)


def test_extract_parses_paddleocr3_nested_result(image_tmp_path: Path) -> None:
    """Parse PaddleOCR 3.x rec_texts, rec_scores, and rec_polys output."""
    image_path = image_tmp_path / "paddle3.png"
    create_text_image(image_path, "Line one")
    engine = FakeOCREngine(
        [
            {
                "res": {
                    "rec_texts": ["Line two", "Line one"],
                    "rec_scores": [0.82, 0.93],
                    "rec_polys": [
                        [[20, 90], [200, 90], [200, 120], [20, 120]],
                        [[20, 50], [200, 50], [200, 80], [20, 80]],
                    ],
                }
            }
        ]
    )

    page = extract_image_text(image_path, "doc-paddle3", "paddle3.png", ocr_engine=engine)

    assert page.text == "Line one\nLine two"
    assert page.average_confidence == pytest.approx((0.82 + 0.93) / 2)


def test_extract_keeps_legacy_list_ocr_format_compatibility(image_tmp_path: Path) -> None:
    """Keep controlled compatibility for legacy PaddleOCR list results."""
    image_path = image_tmp_path / "legacy.png"
    create_text_image(image_path, "Legacy text")
    engine = FakeOCREngine(
        [
            [
                [[20, 50], [200, 50], [200, 80], [20, 80]],
                ("Legacy text", 0.88),
            ]
        ]
    )

    page = extract_image_text(image_path, "doc-legacy", "legacy.png", ocr_engine=engine)

    assert page.text == "Legacy text"
    assert page.average_confidence == pytest.approx(0.88)


def test_extract_rgba_png_with_mocked_ocr(image_tmp_path: Path) -> None:
    """Process RGBA PNG files after RGB conversion."""
    image_path = image_tmp_path / "rgba.png"
    image = Image.new("RGBA", (420, 140), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.text((20, 50), "RGBA text", fill=(0, 0, 0, 255))
    image.save(image_path)
    engine = FakeOCREngine(
        [
            [
                [[20, 50], [180, 50], [180, 80], [20, 80]],
                ("RGBA text", 0.90),
            ]
        ]
    )

    page = extract_image_text(image_path, "doc-rgba", "rgba.png", ocr_engine=engine)

    assert page.text == "RGBA text"
    assert page.extraction_method == "ocr"


def test_extract_grayscale_png_with_mocked_ocr(image_tmp_path: Path) -> None:
    """Process grayscale PNG files after RGB conversion."""
    image_path = image_tmp_path / "grayscale.png"
    image = Image.new("L", (420, 140), 255)
    draw = ImageDraw.Draw(image)
    draw.text((20, 50), "Gray text", fill=0)
    image.save(image_path)
    engine = FakeOCREngine(
        [
            [
                [[20, 50], [180, 50], [180, 80], [20, 80]],
                ("Gray text", 0.89),
            ]
        ]
    )

    page = extract_image_text(
        image_path,
        "doc-grayscale",
        "grayscale.png",
        ocr_engine=engine,
    )

    assert page.text == "Gray text"
    assert page.extraction_method == "ocr"


def test_extract_english_text_jpg_with_mocked_ocr(image_tmp_path: Path) -> None:
    """Return OCR output for a JPG containing English text."""
    image_path = image_tmp_path / "english.jpg"
    create_text_image(image_path, "English document", image_format="JPEG")
    engine = FakeOCREngine(
        [
            [
                [[20, 50], [250, 50], [250, 80], [20, 80]],
                ("English document", 0.91),
            ]
        ]
    )

    page = extract_image_text(image_path, "doc-2", "english.jpg", ocr_engine=engine)

    assert page.text == "English document"
    assert page.detected_language == "en"
    assert page.warnings == []


def test_extract_empty_image_returns_warning(image_tmp_path: Path) -> None:
    """Return an empty OCR result with a warning for blank images."""
    image_path = image_tmp_path / "blank.png"
    Image.new("RGB", (200, 120), "white").save(image_path)

    page = extract_image_text(image_path, "doc-3", "blank.png", ocr_engine=FakeOCREngine([]))

    assert page.text == ""
    assert page.character_count == 0
    assert page.requires_ocr is False
    assert "OCR metni bulunamadi." in page.warnings


def test_preprocess_rejects_too_small_image(image_tmp_path: Path) -> None:
    """Reject images that are too small for reliable OCR."""
    image_path = image_tmp_path / "tiny.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    with pytest.raises(ImageTooSmallError):
        preprocess_image_for_ocr(image_path)


def test_preprocess_rejects_corrupt_image(image_tmp_path: Path) -> None:
    """Raise a controlled error for corrupt image files."""
    image_path = image_tmp_path / "corrupt.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(InvalidImageError):
        preprocess_image_for_ocr(image_path)


def test_preprocess_applies_exif_orientation_and_rgb(image_tmp_path: Path) -> None:
    """Apply EXIF orientation and convert image data to RGB."""
    image_path = image_tmp_path / "rotated.jpg"
    image = Image.new("RGB", (80, 160), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif)

    processed = preprocess_image_for_ocr(image_path)
    expected = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    assert processed.shape[0] == expected.height
    assert processed.shape[1] == expected.width
    assert processed.shape[2] == 3


def test_ocr_service_raises_controlled_error_when_engine_fails(image_tmp_path: Path) -> None:
    """Raise a controlled error when the OCR backend fails."""
    image_path = image_tmp_path / "ocr-failure.png"
    create_text_image(image_path, "Failure")

    with pytest.raises(OCRUnavailableError):
        extract_image_text(
            image_path,
            "doc-4",
            "ocr-failure.png",
            ocr_engine=FakeOCREngine(should_fail=True),
        )


def test_get_ocr_engine_raises_controlled_error_when_paddleocr_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a controlled error when PaddleOCR is unavailable."""
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "paddleocr":
            raise ImportError("paddleocr is intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(ocr_service, "_OCR_ENGINE", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OCRUnavailableError):
        get_ocr_engine()


def test_get_ocr_engine_initializes_paddleocr_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialize PaddleOCR lazily and reuse the same engine instance."""
    init_calls: list[dict[str, object]] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs: object) -> None:
            init_calls.append(kwargs)

    monkeypatch.setattr(ocr_service, "_OCR_ENGINE", None)
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=FakePaddleOCR),
    )

    first_engine = get_ocr_engine()
    second_engine = get_ocr_engine()

    assert first_engine is second_engine
    assert len(init_calls) == 1
    assert init_calls[0]["lang"] == "en"
    assert init_calls[0]["device"] == "cpu"
    assert init_calls[0]["use_doc_orientation_classify"] is False
    assert init_calls[0]["use_doc_unwarping"] is False
    assert init_calls[0]["use_textline_orientation"] is False
    assert "use_gpu" not in init_calls[0]
    assert "show_log" not in init_calls[0]


def test_get_ocr_engine_rejects_unsupported_latin_lang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported PaddleOCR language settings before model initialization."""
    class FakePaddleOCR:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("PaddleOCR should not initialize for invalid lang")

    monkeypatch.setattr(ocr_service, "_OCR_ENGINE", None)
    monkeypatch.setattr(ocr_service.settings, "ocr_lang", "latin")
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=FakePaddleOCR),
    )

    with pytest.raises(OCRUnavailableError, match="Gecersiz OCR dili ayari"):
        get_ocr_engine()


def test_get_ocr_engine_rejects_invalid_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid OCR device settings before model initialization."""
    class FakePaddleOCR:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("PaddleOCR should not initialize for invalid device")

    monkeypatch.setattr(ocr_service, "_OCR_ENGINE", None)
    monkeypatch.setattr(ocr_service.settings, "ocr_device", "invalid-device")
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=FakePaddleOCR),
    )

    with pytest.raises(OCRUnavailableError, match="Gecersiz OCR cihaz ayari"):
        get_ocr_engine()
