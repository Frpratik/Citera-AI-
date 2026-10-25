import pytest

from app.core.errors import PDFProcessingError
from app.ingestion.extract import clean_text, extract_pages
from tests.pdf_factory import make_pdf


def test_clean_text_joins_hyphenation_and_collapses_whitespace():
    raw = "exami-\nnation   fee\r\n\r\n\r\n\r\nis   due"
    assert clean_text(raw) == "examination fee\n\nis due"


def test_extract_pages_numbers_pages_from_one(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["first page text", "second page text"]))
    pages = extract_pages(path, max_pages=10)
    assert [p.number for p in pages] == [1, 2]
    assert "second page" in pages[1].text


def test_extract_skips_blank_pages_but_keeps_real_numbers(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["", "real content"]))
    pages = extract_pages(path, max_pages=10)
    assert [p.number for p in pages] == [2]


def test_extract_rejects_pdf_without_text(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["", ""]))
    with pytest.raises(PDFProcessingError, match="No text"):
        extract_pages(path, max_pages=10)


def test_extract_rejects_too_many_pages(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["a", "b", "c"]))
    with pytest.raises(PDFProcessingError, match="too many pages"):
        extract_pages(path, max_pages=2)


def test_extract_rejects_corrupted_pdf(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4 garbage that is not a real pdf")
    with pytest.raises(PDFProcessingError):
        extract_pages(path, max_pages=10)
