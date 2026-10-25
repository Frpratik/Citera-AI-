from dataclasses import dataclass
from datetime import datetime

from psycopg_pool import ConnectionPool


@dataclass(frozen=True)
class MessageRecord:
    role: str  # "user" or "assistant"
    content: str
    answer_type: str | None
    created_at: datetime


class MessageRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def add(self, owner_id: str, role: str, content: str, answer_type: str | None = None) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO messages (owner_id, role, content, answer_type) VALUES (%s, %s, %s, %s)",
                (owner_id, role, content, answer_type),
            )

    def add_exchange(self, owner_id: str, question: str, answer: str, answer_type: str) -> None:
        """Save the question and its answer together, so history never holds half a pair."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO messages (owner_id, role, content, answer_type) VALUES (%s, %s, %s, %s)",
                [(owner_id, "user", question, None), (owner_id, "assistant", answer, answer_type)],
            )

    def recent(self, owner_id: str, limit: int) -> list[MessageRecord]:
        """The latest `limit` messages, oldest first."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT role, content, answer_type, created_at FROM messages "
                "WHERE owner_id = %s ORDER BY id DESC LIMIT %s",
                (owner_id, limit),
            ).fetchall()
        return [MessageRecord(**row) for row in reversed(rows)]

    def clear(self, owner_id: str) -> int:
        with self._pool.connection() as conn:
            cur = conn.execute("DELETE FROM messages WHERE owner_id = %s", (owner_id,))
        return cur.rowcount
