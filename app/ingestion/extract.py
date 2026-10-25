import logging
import re
from pathlib import Path

from pypdf import PdfReader

from app.core.errors import PDFProcessingError
from app.ingestion.models import Page

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Light cleanup that makes chunks more readable without changing meaning."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "exami-\nnation" -> "examination"
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(path: Path, max_pages: int) -> list[Page]:
    """Extract cleaned text for each page that has any. Page numbers are 1-based."""
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise PDFProcessingError("This PDF is password-protected.")
        if len(reader.pages) > max_pages:
            raise PDFProcessingError(f"This PDF has too many pages (limit is {max_pages}).")
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except PDFProcessingError:
        raise
    except Exception as exc:  # pypdf raises many different error types
        logger.warning("PDF could not be parsed: %s", type(exc).__name__)
        raise PDFProcessingError(
            "This PDF could not be read. It may be corrupted."
        ) from exc

    pages = [
        Page(number=i, text=cleaned)
        for i, raw in enumerate(raw_pages, start=1)
        if (cleaned := clean_text(raw))
    ]
    if not pages:
        raise PDFProcessingError(
            "No text could be extracted. The PDF may be scanned images (OCR is not supported)."
        )
    return pages
