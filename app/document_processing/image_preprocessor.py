"""Image preparation helpers for OCR."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from app.core.config import settings


class ImagePreprocessingError(Exception):
    """Raised when an image cannot be prepared for OCR."""


class InvalidImageError(ImagePreprocessingError):
    """Raised when an image file cannot be opened."""


class ImageTooSmallError(ImagePreprocessingError):
    """Raised when an image is too small for reliable OCR."""


def preprocess_image_for_ocr(image_path: str | Path) -> np.ndarray:
    """Load an image, normalize orientation, and return an RGB array for OCR."""
    path = Path(image_path)
    if not path.exists():
        raise InvalidImageError("Goruntu dosyasi bulunamadi.")

    try:
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image)
            rgb_image = normalized.convert("RGB")
            _validate_image_size(rgb_image)
            enhanced = _enhance_for_ocr(rgb_image)
            return np.asarray(enhanced)
    except UnidentifiedImageError as exc:
        raise InvalidImageError("Goruntu dosyasi okunamadi veya bozuk.") from exc
    except OSError as exc:
        raise InvalidImageError("Goruntu dosyasi okunamadi.") from exc


def _validate_image_size(image: Image.Image) -> None:
    width, height = image.size
    if width < settings.ocr_min_image_width or height < settings.ocr_min_image_height:
        raise ImageTooSmallError(
            "Goruntu OCR icin cok kucuk. Daha yuksek cozunurluklu bir belge yukleyin."
        )


def _enhance_for_ocr(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(grayscale)
    enhanced = ImageEnhance.Contrast(contrasted).enhance(1.25)
    return enhanced.convert("RGB")
