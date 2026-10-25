"""Ingestion in two halves.

  accept()   quick, runs inside the request: check and save the file, create the document
             (status "processing") and queue a job. The user gets an answer straight away.
  process()  slow, runs in a background worker: read, chunk, embed, store, mark ready.

A document row exists from the start with a status, so a failure is visible to the user
("failed" plus a reason) instead of silently vanishing.
"""

import logging
from pathlib import Path
from typing import BinaryIO

from app.core.config import Settings
from app.core.errors import AppError, NotFoundError, ValidationFailedError
from app.embeddings.service import Embedder
from app.ingestion.chunking import chunk_pages
from app.ingestion.extract import extract_pages
from app.ingestion.upload import sanitize_filename, save_upload
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRecord, DocumentRepository
from app.repositories.jobs import JobRepository

logger = logging.getLogger(__name__)

MISSING_FILE_MESSAGE = "The original file is missing. Please upload it again."


class IngestionService:
    def __init__(
        self,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        jobs: JobRepository,
        embedder: Embedder,
        settings: Settings,
    ):
        self._documents = documents
        self._chunks = chunks
        self._jobs = jobs
        self._embedder = embedder
        self._settings = settings

    def warm_up(self) -> None:
        self._embedder.warm_up()

    def _file_path(self, document_id: str) -> Path:
        # Named by server-generated UUID, never by anything the user typed.
        return self._settings.data_dir / "uploads" / f"{document_id}.pdf"

    # ------------------------------------------------------------------ queries
    def get(self, owner_id: str, document_id: str) -> DocumentRecord:
        record = self._documents.get(owner_id, document_id)
        if record is None:
            # Same answer whether it does not exist or belongs to someone else.
            raise NotFoundError("Document not found.")
        return record

    def list(self, owner_id: str) -> list[DocumentRecord]:
        return self._documents.list_for_owner(owner_id)

    # ------------------------------------------------------------------ the quick half
    def _create_and_save(self, owner_id: str, filename: str | None, fileobj: BinaryIO) -> DocumentRecord:
        display_name = sanitize_filename(filename)  # raises InvalidUploadError
        record = self._documents.create(owner_id, display_name)
        try:
            save_upload(fileobj, self._file_path(record.id), self._settings.max_upload_bytes)
        except Exception:
            self._documents.delete(owner_id, record.id)  # a rejected upload leaves no trace
            raise
        return record

    def accept(self, owner_id: str, filename: str | None, fileobj: BinaryIO) -> DocumentRecord:
        """Save the upload and queue its processing. Returns at once with status "processing"."""
        record = self._create_and_save(owner_id, filename, fileobj)
        self._jobs.enqueue(record.id, self._settings.job_max_attempts)
        return record

    def ingest_now(self, owner_id: str, filename: str | None, fileobj: BinaryIO) -> DocumentRecord:
        """Save and process in one go, without the queue. For scripts and tests."""
        record = self._create_and_save(owner_id, filename, fileobj)
        self.process(record.id)
        return self.get(owner_id, record.id)

    def reprocess(self, owner_id: str, document_id: str) -> DocumentRecord:
        """Try a failed document again."""
        record = self.get(owner_id, document_id)
        if record.status != "failed":
            raise ValidationFailedError("Only documents that failed can be reprocessed.")
        if not self._file_path(record.id).exists():
            self.mark_failed(record.id, MISSING_FILE_MESSAGE)
            return self.get(owner_id, document_id)
        self._chunks.delete_for_document(record.id)
        self._documents.mark_processing(record.id)
        self._jobs.enqueue(record.id, self._settings.job_max_attempts)
        return self.get(owner_id, document_id)

    def delete(self, owner_id: str, document_id: str) -> None:
        self.get(owner_id, document_id)  # 404 if missing or not yours
        # The row first: search only sees documents listed in the database (its chunks and
        # job go with it), so the document stops appearing in answers immediately.
        self._documents.delete(owner_id, document_id)
        try:
            self._file_path(document_id).unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not delete the uploaded file for %s", document_id)

    # ------------------------------------------------------------------ the slow half
    def process(self, document_id: str) -> None:
        """Read, chunk, embed and store a document, then mark it ready. Called by the worker.

        Two kinds of failure are treated differently:
          * the FILE is the problem (corrupt, no text, too long): retrying cannot help, so
            the document is marked failed with a clear reason and we return normally;
          * anything else (database or model trouble): the error propagates, and the
            worker retries the job later.
        Safe to run twice for the same document: old chunks are removed first.
        """
        record = self._documents.get_by_id(document_id)
        if record is None:  # deleted while it waited in the queue
            logger.info("Document %s no longer exists; nothing to process", document_id)
            return

        path = self._file_path(document_id)
        if not path.exists():
            self.mark_failed(document_id, MISSING_FILE_MESSAGE)
            return

        s = self._settings
        try:
            pages = extract_pages(path, s.max_pdf_pages)
            chunks = chunk_pages(pages, document_id, s.chunk_size, s.chunk_overlap)
        except AppError as exc:
            self.mark_failed(document_id, exc.message)
            return

        vectors = self._embedder.embed_documents([c.text for c in chunks])
        self._chunks.delete_for_document(document_id)
        self._chunks.add(document_id, chunks, vectors)
        self._documents.mark_ready(document_id, len(pages), len(chunks))
        logger.info("Document %s ready (%d pages, %d chunks)", document_id, len(pages), len(chunks))

    def mark_failed(self, document_id: str, message: str) -> None:
        self._chunks.delete_for_document(document_id)  # drop partial output
        self._documents.mark_failed(document_id, message)
