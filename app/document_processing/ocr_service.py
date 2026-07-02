"""OCR service for image documents."""

from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from app.core.config import settings
from app.document_processing.image_preprocessor import (
    ImagePreprocessingError,
    preprocess_image_for_ocr,
)
from app.document_processing.models import ExtractedPage
from app.document_processing.text_cleaner import clean_extracted_text

OCR_EXTRACTION_METHOD = "ocr"
OCR_PDF_EXTRACTION_METHOD = "ocr_pdf"
UNSUPPORTED_OCR_LANGS = {"latin"}
SUPPORTED_OCR_DEVICE_PREFIXES = ("cpu", "gpu")
_OCR_ENGINE: Any | None = None
_OCR_ENGINE_LOCK = Lock()


class OCRServiceError(Exception):
    """Raised when OCR cannot be completed safely."""


class OCRUnavailableError(OCRServiceError):
    """Raised when the OCR engine cannot be initialized or used."""


def extract_image_text(
    image_path: str | Path,
    document_id: str,
    filename: str,
    ocr_engine: Any | None = None,
) -> ExtractedPage:
    """Extract OCR text from a single image document."""
    try:
        image = preprocess_image_for_ocr(image_path)
    except ImagePreprocessingError:
        raise

    engine = ocr_engine if ocr_engine is not None else get_ocr_engine()
    return extract_image_array_text(
        image=image,
        document_id=document_id,
        filename=filename,
        page_number=1,
        extraction_method=OCR_EXTRACTION_METHOD,
        ocr_engine=engine,
    )


def extract_image_array_text(
    image: np.ndarray,
    document_id: str,
    filename: str,
    page_number: int,
    extraction_method: str = OCR_EXTRACTION_METHOD,
    ocr_engine: Any | None = None,
) -> ExtractedPage:
    """Extract OCR text from an in-memory image array."""
    engine = ocr_engine if ocr_engine is not None else get_ocr_engine()
    try:
        raw_result = _run_ocr(engine, image)
    except Exception as exc:
        raise OCRUnavailableError("OCR servisi calistirilamadi.") from exc

    lines = _extract_ocr_lines(raw_result)
    ordered_lines = sorted(lines, key=lambda line: (_line_top(line), _line_left(line)))
    text = clean_extracted_text("\n".join(line["text"] for line in ordered_lines))
    confidences = [
        float(line["confidence"])
        for line in ordered_lines
        if line.get("confidence") is not None
    ]
    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    return ExtractedPage(
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        text=text,
        extraction_method=extraction_method,
        character_count=len(text),
        requires_ocr=False,
        detected_language=_get_configured_ocr_lang(),
        average_confidence=average_confidence,
        warnings=_build_warnings(text, average_confidence),
    )


def get_ocr_engine() -> Any:
    """Return a lazily initialized PaddleOCR engine."""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise OCRUnavailableError("PaddleOCR kurulumu bulunamadi.") from exc

            _OCR_ENGINE = _initialize_paddle_ocr(PaddleOCR)
        return _OCR_ENGINE


def _initialize_paddle_ocr(paddle_ocr_class: Any) -> Any:
    ocr_lang = _get_configured_ocr_lang()
    ocr_device = _get_configured_ocr_device()
    try:
        return paddle_ocr_class(
            lang=ocr_lang,
            device=ocr_device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:
        raise OCRUnavailableError("PaddleOCR baslatilamadi.") from exc


def _get_configured_ocr_lang() -> str:
    ocr_lang = settings.ocr_lang.strip().lower()
    if not ocr_lang or ocr_lang in UNSUPPORTED_OCR_LANGS:
        raise OCRUnavailableError("Gecersiz OCR dili ayari.")
    return ocr_lang


def _get_configured_ocr_device() -> str:
    ocr_device = settings.ocr_device.strip().lower()
    if not ocr_device or not ocr_device.startswith(SUPPORTED_OCR_DEVICE_PREFIXES):
        raise OCRUnavailableError("Gecersiz OCR cihaz ayari.")
    return ocr_device


def _run_ocr(engine: Any, image: np.ndarray) -> Any:
    if hasattr(engine, "predict"):
        return engine.predict(image)
    raise OCRUnavailableError("OCR motoru beklenen arayuzu saglamiyor.")


def _extract_ocr_lines(raw_result: Any) -> list[dict[str, Any]]:
    if raw_result is None:
        return []
    if hasattr(raw_result, "to_dict"):
        return _extract_ocr_lines(raw_result.to_dict())
    if hasattr(raw_result, "json"):
        return _extract_ocr_lines(raw_result.json)
    if isinstance(raw_result, dict):
        return _extract_lines_from_dict(raw_result)
    if isinstance(raw_result, list):
        lines: list[dict[str, Any]] = []
        for item in raw_result:
            if isinstance(item, dict):
                lines.extend(_extract_lines_from_dict(item))
            elif _looks_like_ocr_line(item):
                line = _line_from_sequence(item)
                if line is not None:
                    lines.append(line)
            elif isinstance(item, list):
                lines.extend(_extract_ocr_lines(item))
        return lines
    return []


def _extract_lines_from_dict(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("res"), dict):
        payload = payload["res"]
    texts = payload.get("rec_texts") or payload.get("texts") or []
    scores = payload.get("rec_scores") or payload.get("scores") or []
    boxes = (
        payload.get("rec_polys")
        or payload.get("dt_polys")
        or payload.get("rec_boxes")
        or payload.get("boxes")
        or []
    )
    lines: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        if not text:
            continue
        lines.append(
            {
                "text": str(text),
                "confidence": scores[index] if index < len(scores) else None,
                "box": boxes[index] if index < len(boxes) else None,
            }
        )
    return lines


def _looks_like_ocr_line(item: Any) -> bool:
    return (
        isinstance(item, (list, tuple))
        and len(item) >= 2
        and isinstance(item[1], (list, tuple))
        and len(item[1]) >= 1
        and isinstance(item[1][0], str)
    )


def _line_from_sequence(item: Any) -> dict[str, Any] | None:
    text_info = item[1]
    text = str(text_info[0])
    if not text:
        return None
    confidence = text_info[1] if len(text_info) > 1 else None
    return {"text": text, "confidence": confidence, "box": item[0]}


def _line_top(line: dict[str, Any]) -> float:
    points = _box_points(line.get("box"))
    if not points:
        return 0.0
    return min(point[1] for point in points)


def _line_left(line: dict[str, Any]) -> float:
    points = _box_points(line.get("box"))
    if not points:
        return 0.0
    return min(point[0] for point in points)


def _box_points(box: Any) -> list[tuple[float, float]]:
    if box is None:
        return []
    array = np.asarray(box).reshape(-1, 2)
    return [(float(point[0]), float(point[1])) for point in array]


def _build_warnings(text: str, average_confidence: float | None) -> list[str]:
    warnings: list[str] = []
    if not text:
        warnings.append("OCR metni bulunamadi.")
    if (
        average_confidence is not None
        and average_confidence < settings.ocr_low_confidence_threshold
    ):
        warnings.append("OCR confidence degeri dusuk.")
    return warnings
