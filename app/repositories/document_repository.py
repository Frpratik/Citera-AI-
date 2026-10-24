"""SQLite storage for document metadata and chat history.

Why SQLite? FAISS only stores vectors. It cannot tell us "which documents does this
user own?", "is this one still processing?" or "what was said earlier in the chat?".
SQLite ships with Python, needs no server, and is plenty for a single-machine app.

Every query that touches user data is filtered by `owner_id`, which is how one
user's documents stay invisible to another.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT NOT NULL,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,
    page_count  INTEGER,
    chunk_count INTEGER,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    answer_type TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_owner ON messages(owner_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    owner_id: str
    filename: str
    status: str  # processing | ready | failed
    page_count: int | None
    chunk_count: int | None
    error: str | None
    created_at: str
    updated_at: str


class DocumentRepository:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A fresh connection per operation is the simplest thread-safe approach
        # (FastAPI runs our sync endpoints in a thread pool).
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(**{key: row[key] for key in row.keys()})

    # ------------------------------------------------------------------ documents
    def create(self, owner_id: str, filename: str) -> DocumentRecord:
        document_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, owner_id, filename, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'processing', ?, ?)",
                (document_id, owner_id, filename, now, now),
            )
        record = self.get(owner_id, document_id)
        assert record is not None
        return record

    def get(self, owner_id: str, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ? AND owner_id = ?", (document_id, owner_id)
            ).fetchone()
        return self._to_record(row) if row else None

    def list_for_owner(self, owner_id: str) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE owner_id = ? ORDER BY created_at, rowid", (owner_id,)
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_ready(self, owner_id: str) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE owner_id = ? AND status = 'ready' "
                "ORDER BY created_at, rowid",
                (owner_id,),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_all(self) -> list[DocumentRecord]:
        """All users' documents. Used only by startup recovery, never by API requests."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
        return [self._to_record(row) for row in rows]

    def count_for_owner(self, owner_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE owner_id = ?", (owner_id,)
            ).fetchone()
        return int(row["n"])

    def update_status(
        self,
        document_id: str,
        status: str,
        *,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, page_count = ?, chunk_count = ?, error = ?, "
                "updated_at = ? WHERE id = ?",
                (status, page_count, chunk_count, error, _now(), document_id),
            )

    def delete(self, owner_id: str, document_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE id = ? AND owner_id = ?", (document_id, owner_id)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ chat history
    def add_message(
        self, owner_id: str, role: str, content: str, answer_type: str | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (owner_id, role, content, answer_type, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (owner_id, role, content, answer_type, _now()),
            )

    def recent_messages(self, owner_id: str, limit: int) -> list[dict]:
        """Newest `limit` messages, returned oldest-first."""
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, answer_type, created_at FROM messages "
                "WHERE owner_id = ? ORDER BY id DESC LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def clear_messages(self, owner_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE owner_id = ?", (owner_id,))
        return cursor.rowcount
