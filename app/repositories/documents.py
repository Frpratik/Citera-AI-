import uuid
from dataclasses import dataclass
from datetime import datetime

from psycopg_pool import ConnectionPool


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    owner_id: str
    filename: str
    status: str  # processing | ready | failed
    page_count: int | None
    chunk_count: int | None
    error: str | None
    created_at: datetime


def _to_record(row: dict) -> DocumentRecord:
    return DocumentRecord(**{**row, "id": str(row["id"])})


class DocumentRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create(self, owner_id: str, filename: str) -> DocumentRecord:
        doc_id = str(uuid.uuid4())
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO documents (id, owner_id, filename, status) "
                "VALUES (%s, %s, %s, 'processing') RETURNING *",
                (doc_id, owner_id, filename),
            ).fetchone()
        return _to_record(row)

    def get_by_id(self, document_id: str) -> DocumentRecord | None:
        """Without an owner check: for the background worker only, never for a request."""
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = %s", (document_id,)).fetchone()
        return _to_record(row) if row else None

    def mark_processing(self, document_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE documents SET status = 'processing', error = NULL WHERE id = %s", (document_id,)
            )

    def get(self, owner_id: str, document_id: str) -> DocumentRecord | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = %s AND owner_id = %s", (document_id, owner_id)
            ).fetchone()
        return _to_record(row) if row else None

    def list_for_owner(self, owner_id: str) -> list[DocumentRecord]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE owner_id = %s ORDER BY created_at DESC", (owner_id,)
            ).fetchall()
        return [_to_record(r) for r in rows]

    def count_ready(self, owner_id: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM documents WHERE owner_id = %s AND status = 'ready'",
                (owner_id,),
            ).fetchone()
        return int(row["n"])

    def mark_ready(self, document_id: str, page_count: int, chunk_count: int) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE documents SET status = 'ready', page_count = %s, chunk_count = %s, "
                "error = NULL WHERE id = %s",
                (page_count, chunk_count, document_id),
            )

    def mark_failed(self, document_id: str, error: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
                (error, document_id),
            )

    def delete(self, owner_id: str, document_id: str) -> bool:
        """Delete a document; its chunks go with it (ON DELETE CASCADE)."""
        with self._pool.connection() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE id = %s AND owner_id = %s", (document_id, owner_id)
            )
        return cur.rowcount > 0
