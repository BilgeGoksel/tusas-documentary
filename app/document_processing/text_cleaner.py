"""Text cleaning helpers for extracted document text."""

import re

_HORIZONTAL_SPACE_PATTERN = re.compile(r"[^\S\r\n]+")
_TOO_MANY_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def clean_extracted_text(text: str) -> str:
    """Reduce noisy whitespace without changing text meaning."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [
        _HORIZONTAL_SPACE_PATTERN.sub(" ", line).strip()
        for line in normalized.split("\n")
    ]
    cleaned = "\n".join(cleaned_lines).strip()
    return _TOO_MANY_BLANK_LINES_PATTERN.sub("\n\n", cleaned)
