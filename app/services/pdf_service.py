"""PDF handling: safe saving, text extraction, cleaning and chunking.

Example of what this module does to one page of text:

    Page 3 text (2,400 characters)
      -> cleaned (extra whitespace, hyphenated line breaks removed)
      -> split into ~1000-character chunks that overlap by ~150 characters
      -> each chunk keeps metadata: document id, document name, page number, chunk id

The overlap means a sentence that falls on a chunk boundary still appears
whole in at least one chunk.
"""

import logging
import re
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.errors import FileTooLargeError, InvalidUploadError
from app.services.ids import check_ids, safe_child_path

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
MAX_DISPLAY_NAME_LENGTH = 120


class PDFProcessingError(Exception):
    """A PDF could not be turned into text. The message is safe to show to the user."""


def sanitize_filename(raw_name: str | None) -> str:
    """Return a safe *display* name for an uploaded file.

    The uploaded name is never used as a path (we store files as `<document_id>.pdf`),
    but it is shown in the UI and in citations, so we still strip folders and
    control characters and require a .pdf extension.
    """
    name = (raw_name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name.lower().endswith(".pdf") or len(name) <= len(".pdf"):
        raise InvalidUploadError("Only PDF files are accepted (the file name must end in .pdf).")
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        name = name[: MAX_DISPLAY_NAME_LENGTH - len(".pdf")].rstrip() + ".pdf"
    return name


def clean_text(text: str) -> str:
    """Light cleanup that makes chunks more readable without changing meaning."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "exami-\nnation" -> "examination"
    text = re.sub(r"[ \t\f\v]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()


class PDFService:
    def __init__(
        self,
        uploads_dir: Path,
        max_upload_bytes: int,
        max_pages: int,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._uploads_dir = uploads_dir
        self._max_upload_bytes = max_upload_bytes
        self._max_pages = max_pages
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    # ------------------------------------------------------------------ files
    def document_path(self, owner_id: str, document_id: str) -> Path:
        check_ids(owner_id, document_id)
        return safe_child_path(self._uploads_dir, owner_id, f"{document_id}.pdf")

    def save_upload(self, owner_id: str, document_id: str, fileobj: BinaryIO) -> Path:
        """Stream the upload to disk, enforcing the size limit and PDF signature."""
        path = self.document_path(owner_id, document_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        total = 0
        first_bytes = b""
        try:
            with open(path, "wb") as out:
                while True:
                    block = fileobj.read(1024 * 1024)
                    if not block:
                        break
                    if total == 0:
                        first_bytes = block[: len(PDF_MAGIC)]
                    total += len(block)
                    if total > self._max_upload_bytes:
                        raise FileTooLargeError(
                            f"File is too large. The limit is {self._max_upload_bytes // (1024 * 1024)} MB."
                        )
                    out.write(block)
            if total == 0:
                raise InvalidUploadError("The uploaded file is empty.")
            if first_bytes != PDF_MAGIC:
                raise InvalidUploadError("The file does not look like a valid PDF.")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def delete_file(self, owner_id: str, document_id: str) -> None:
        self.document_path(owner_id, document_id).unlink(missing_ok=True)

    # ------------------------------------------------------------------ extraction
    def load_pages(self, path: Path) -> list[Document]:
        """Extract one LangChain Document per PDF page using PyPDFLoader."""
        try:
            pages = PyPDFLoader(str(path)).load()
        except Exception as exc:  # pypdf raises many different error types
            logger.warning("PDF could not be parsed: %s", type(exc).__name__)
            raise PDFProcessingError(
                "This PDF could not be read. It may be corrupted or password-protected."
            ) from exc

        if not pages:
            raise PDFProcessingError("This PDF contains no pages.")
        if len(pages) > self._max_pages:
            raise PDFProcessingError(f"This PDF has too many pages (limit is {self._max_pages}).")
        if not any(page.page_content.strip() for page in pages):
            raise PDFProcessingError(
                "No text could be extracted. The PDF may be scanned images (OCR is not supported)."
            )
        return pages

    # ------------------------------------------------------------------ chunking
    def split_pages(
        self, pages: list[Document], document_id: str, document_name: str
    ) -> list[Document]:
        """Clean pages, split them into chunks, and attach metadata to every chunk."""
        cleaned_pages: list[Document] = []
        for index, page in enumerate(pages):
            text = clean_text(page.page_content)
            if not text:
                continue
            # PyPDFLoader numbers pages from 0; humans count from 1.
            page_number = int(page.metadata.get("page", index)) + 1
            cleaned_pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "document_id": document_id,
                        "document_name": document_name,
                        "page": page_number,
                    },
                )
            )

        chunks = self._splitter.split_documents(cleaned_pages)
        if not chunks:
            raise PDFProcessingError("No usable text was found in this PDF.")

        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = f"{document_id}-c{chunk_index}"
        return chunks
