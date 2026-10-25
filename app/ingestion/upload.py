import re
from pathlib import Path
from typing import BinaryIO

from app.core.errors import FileTooLargeError, InvalidUploadError

PDF_MAGIC = b"%PDF-"
MAX_DISPLAY_NAME_LENGTH = 120
READ_BLOCK = 1024 * 1024


def sanitize_filename(raw_name: str | None) -> str:
    """Return a safe *display* name. The name is never used to build a file path."""
    name = (raw_name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name.lower().endswith(".pdf") or len(name) <= len(".pdf"):
        raise InvalidUploadError("Only PDF files are accepted (the file name must end in .pdf).")
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        name = name[: MAX_DISPLAY_NAME_LENGTH - len(".pdf")].rstrip() + ".pdf"
    return name


def save_upload(fileobj: BinaryIO, dest: Path, max_bytes: int) -> int:
    """Stream an upload to `dest`, enforcing the size limit and the PDF signature.

    Reads in blocks so a huge upload is rejected without being held in memory.
    On any failure the partial file is removed. Returns the number of bytes saved.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    head = b""
    try:
        with open(dest, "wb") as out:
            while block := fileobj.read(READ_BLOCK):
                if total == 0:
                    head = block[: len(PDF_MAGIC)]
                total += len(block)
                if total > max_bytes:
                    raise FileTooLargeError(
                        f"File is too large. The limit is {max_bytes // (1024 * 1024)} MB."
                    )
                out.write(block)
        if total == 0:
            raise InvalidUploadError("The uploaded file is empty.")
        if head != PDF_MAGIC:
            raise InvalidUploadError("The file does not look like a valid PDF.")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return total
