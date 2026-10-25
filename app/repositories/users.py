import uuid
from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg_pool import ConnectionPool

from app.core.errors import EmailTakenError


@dataclass(frozen=True)
class UserRecord:
    id: str
    email: str
    created_at: datetime


def _to_record(row: dict) -> UserRecord:
    return UserRecord(id=str(row["id"]), email=row["email"], created_at=row["created_at"])


class UserRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def create(self, email: str, password_hash: str) -> UserRecord:
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "INSERT INTO users (id, email, password_hash) VALUES (%s, %s, %s) "
                    "RETURNING id, email, created_at",
                    (str(uuid.uuid4()), email, password_hash),
                ).fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise EmailTakenError("An account with this email already exists.") from exc
        return _to_record(row)

    def get(self, user_id: str) -> UserRecord | None:
        try:
            uuid.UUID(user_id)  # a malformed id (from a forged token) must not reach the query
        except ValueError:
            return None
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, email, created_at FROM users WHERE id = %s", (user_id,)
            ).fetchone()
        return _to_record(row) if row else None

    def get_credentials(self, email: str) -> tuple[UserRecord, str] | None:
        """The user and their password hash, for login only."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, email, created_at, password_hash FROM users WHERE lower(email) = lower(%s)",
                (email,),
            ).fetchone()
        return (_to_record(row), row["password_hash"]) if row else None
