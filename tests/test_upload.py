import io

import pytest

from app.core.errors import FileTooLargeError, InvalidUploadError
from app.ingestion.upload import sanitize_filename, save_upload
from tests.pdf_factory import make_pdf


def test_sanitize_strips_folders_and_control_chars():
    assert sanitize_filename("C:\\docs\\..\\my\x00 report.PDF") == "my report.PDF"


@pytest.mark.parametrize("bad", [None, "", "notes.txt", ".pdf", "evil.pdf.exe"])
def test_sanitize_rejects_non_pdf_names(bad):
    with pytest.raises(InvalidUploadError):
        sanitize_filename(bad)


def test_sanitize_truncates_long_names():
    name = sanitize_filename("a" * 300 + ".pdf")
    assert len(name) == 120 and name.endswith(".pdf")


def test_save_valid_pdf(tmp_path):
    dest = tmp_path / "sub" / "doc.pdf"
    size = save_upload(io.BytesIO(make_pdf(["hello"])), dest, max_bytes=1_000_000)
    assert dest.exists() and size == dest.stat().st_size


def test_save_rejects_wrong_signature_and_cleans_up(tmp_path):
    dest = tmp_path / "doc.pdf"
    with pytest.raises(InvalidUploadError):
        save_upload(io.BytesIO(b"MZ not a pdf"), dest, max_bytes=1_000_000)
    assert not dest.exists()


def test_save_rejects_empty(tmp_path):
    with pytest.raises(InvalidUploadError):
        save_upload(io.BytesIO(b""), tmp_path / "doc.pdf", max_bytes=1_000_000)


def test_save_rejects_too_large_and_cleans_up(tmp_path):
    dest = tmp_path / "doc.pdf"
    with pytest.raises(FileTooLargeError):
        save_upload(io.BytesIO(b"%PDF-" + b"x" * 5000), dest, max_bytes=1000)
    assert not dest.exists()
