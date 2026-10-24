"""PDF extraction, cleaning, chunking and filename safety."""

import io

import pytest

from app.core.errors import FileTooLargeError, InvalidUploadError
from app.services.pdf_service import PDFProcessingError, PDFService, clean_text, sanitize_filename
from tests.helpers import ATTENDANCE_TEXT, FEES_TEXT, make_pdf

OWNER = "owner-test-0001"
DOC_ID = "a" * 32


@pytest.fixture
def pdf_service(tmp_path) -> PDFService:
    return PDFService(tmp_path / "uploads", max_upload_bytes=1024 * 1024, max_pages=50,
                      chunk_size=1000, chunk_overlap=150)


def _save(pdf_service, data: bytes):
    return pdf_service.save_upload(OWNER, DOC_ID, io.BytesIO(data))


def test_extracts_text_per_page_with_page_numbers(pdf_service):
    path = _save(pdf_service, make_pdf([ATTENDANCE_TEXT, FEES_TEXT]))
    pages = pdf_service.load_pages(path)
    assert len(pages) == 2
    assert "75%" in pages[0].page_content
    assert "tuition" in pages[1].page_content


def test_chunks_carry_document_metadata(pdf_service):
    path = _save(pdf_service, make_pdf([ATTENDANCE_TEXT, FEES_TEXT]))
    chunks = pdf_service.split_pages(pdf_service.load_pages(path), DOC_ID, "rules.pdf")
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["document_id"] == DOC_ID
        assert chunk.metadata["document_name"] == "rules.pdf"
        assert chunk.metadata["page"] in (1, 2)  # human page numbers start at 1
        assert chunk.metadata["chunk_id"].startswith(DOC_ID)
    assert chunks[0].metadata["page"] == 1


def test_long_page_is_split_into_overlapping_chunks(pdf_service):
    long_text = " ".join(f"Sentence number {i} about hostel rules." for i in range(120))
    path = _save(pdf_service, make_pdf([long_text]))
    chunks = pdf_service.split_pages(pdf_service.load_pages(path), DOC_ID, "hostel.pdf")
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 1000 for c in chunks)


def test_rejects_non_pdf_content(pdf_service):
    with pytest.raises(InvalidUploadError):
        _save(pdf_service, b"this is definitely not a pdf")


def test_rejects_empty_file(pdf_service):
    with pytest.raises(InvalidUploadError):
        _save(pdf_service, b"")


def test_rejects_oversized_file_and_removes_partial_file(pdf_service):
    with pytest.raises(FileTooLargeError):
        _save(pdf_service, b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024))
    assert not pdf_service.document_path(OWNER, DOC_ID).exists()


def test_malformed_pdf_gives_friendly_error(pdf_service):
    path = _save(pdf_service, b"%PDF-1.4\nthis is broken\n%%EOF")
    with pytest.raises(PDFProcessingError):
        pdf_service.load_pages(path)


def test_pdf_without_text_is_rejected(pdf_service):
    path = _save(pdf_service, make_pdf(["   "]))
    with pytest.raises(PDFProcessingError):
        pdf_service.load_pages(path)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("rules.pdf", "rules.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\fees.PDF", "fees.PDF"),
        ("bad\x00name\n.pdf", "badname.pdf"),
    ],
)
def test_sanitize_filename_strips_paths_and_control_chars(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", ["notes.txt", "", None, ".pdf", "script.pdf.exe"])
def test_sanitize_filename_rejects_non_pdf(raw):
    with pytest.raises(InvalidUploadError):
        sanitize_filename(raw)


def test_clean_text_joins_hyphenated_words_and_collapses_whitespace():
    assert clean_text("exami-\nnation   fee\n\n\n\nrules") == "examination fee\n\nrules"
