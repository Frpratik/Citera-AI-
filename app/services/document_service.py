"""Document lifecycle: upload -> process -> index -> list -> reprocess -> delete.

Processing is synchronous in this version: the upload request waits until the PDF is
indexed. That is simple and fine for normal college PDFs (a few seconds). Move
processing to a background worker (e.g. Celery/RQ, or FastAPI BackgroundTasks for
very light loads) when PDFs are large, many users upload at once, or requests start
hitting timeouts. The `status` column and status endpoint are already in place for that.
"""

import logging
from typing import BinaryIO

from app.core.errors import DocumentLimitError, DocumentNotFoundError
from app.repositories.document_repository import DocumentRecord, DocumentRepository
from app.services.pdf_service import PDFProcessingError, PDFService, sanitize_filename
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        pdf_service: PDFService,
        vector_store: VectorStoreService,
        max_documents_per_user: int,
    ):
        self._repository = repository
        self._pdf = pdf_service
        self._vector_store = vector_store
        self._max_documents = max_documents_per_user

    # ------------------------------------------------------------------ queries
    def get(self, owner_id: str, document_id: str) -> DocumentRecord:
        record = self._repository.get(owner_id, document_id)
        if record is None:
            # Same error whether it does not exist or belongs to someone else.
            raise DocumentNotFoundError("Document not found.")
        return record

    def list(self, owner_id: str) -> list[DocumentRecord]:
        return self._repository.list_for_owner(owner_id)

    # ------------------------------------------------------------------ commands
    def upload(self, owner_id: str, filename: str | None, fileobj: BinaryIO) -> DocumentRecord:
        display_name = sanitize_filename(filename)  # raises InvalidUploadError
        if self._repository.count_for_owner(owner_id) >= self._max_documents:
            raise DocumentLimitError(
                f"You can keep at most {self._max_documents} documents. Delete one first."
            )

        record = self._repository.create(owner_id, display_name)
        try:
            self._pdf.save_upload(owner_id, record.id, fileobj)  # validates size + signature
        except Exception:
            self._repository.delete(owner_id, record.id)
            raise
        return self._process(record)

    def reprocess(self, owner_id: str, document_id: str) -> DocumentRecord:
        record = self.get(owner_id, document_id)
        if not self._pdf.document_path(owner_id, document_id).exists():
            self._repository.update_status(
                document_id, "failed", error="The original file is missing. Please upload it again."
            )
            return self.get(owner_id, document_id)
        self._vector_store.delete_index(owner_id, document_id)
        self._repository.update_status(document_id, "processing")
        return self._process(record)

    def delete(self, owner_id: str, document_id: str) -> None:
        self.get(owner_id, document_id)  # 404 if missing / not yours
        # Remove the metadata row FIRST: retrieval only searches documents listed in
        # the database, so the document stops appearing in answers immediately,
        # even if deleting files below were to fail.
        self._repository.delete(owner_id, document_id)
        self._vector_store.delete_index(owner_id, document_id)
        try:
            self._pdf.delete_file(owner_id, document_id)
        except OSError:
            logger.exception("Could not delete uploaded file for %s", document_id)

    def recover_after_restart(self) -> None:
        """Fix up documents whose state does not match the disk after a crash/restart."""
        for record in self._repository.list_all():
            if record.status == "processing":
                self._repository.update_status(
                    record.id,
                    "failed",
                    error="Processing was interrupted. Use Reprocess to try again.",
                )
            elif record.status == "ready" and not self._vector_store.index_exists(
                record.owner_id, record.id
            ):
                self._repository.update_status(
                    record.id, "failed", error="Index files are missing. Use Reprocess to rebuild."
                )

    # ------------------------------------------------------------------ internals
    def _process(self, record: DocumentRecord) -> DocumentRecord:
        """PDF -> pages -> cleaned chunks -> embeddings -> FAISS index."""
        try:
            path = self._pdf.document_path(record.owner_id, record.id)
            pages = self._pdf.load_pages(path)
            chunks = self._pdf.split_pages(pages, record.id, record.filename)
            self._vector_store.build_index(record.owner_id, record.id, chunks)
            self._repository.update_status(
                record.id, "ready", page_count=len(pages), chunk_count=len(chunks)
            )
            logger.info("Document %s ready (%d pages, %d chunks)", record.id, len(pages), len(chunks))
        except PDFProcessingError as exc:
            self._fail(record, str(exc))
        except Exception:
            logger.exception("Unexpected error while processing document %s", record.id)
            self._fail(
                record,
                "Something went wrong while processing this PDF. Try Reprocess, or upload it again.",
            )
        return self.get(record.owner_id, record.id)

    def _fail(self, record: DocumentRecord, message: str) -> None:
        self._vector_store.delete_index(record.owner_id, record.id)  # drop partial output
        self._repository.update_status(record.id, "failed", error=message)
