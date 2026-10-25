from dataclasses import dataclass

from psycopg_pool import ConnectionPool


@dataclass(frozen=True)
class Job:
    id: int
    document_id: str
    attempts: int  # including the current one
    max_attempts: int


class JobRepository:
    """The queue. Several workers can use it at the same time without picking the same job."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def enqueue(self, document_id: str, max_attempts: int = 3) -> int | None:
        """Add a job. Returns its id, or None if this document already has a waiting or
        running job (the database refuses a second one, so double clicks are harmless)."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO ingest_jobs (document_id, max_attempts) VALUES (%s, %s) "
                "ON CONFLICT (document_id) WHERE status IN ('queued', 'running') DO NOTHING "
                "RETURNING id",
                (document_id, max_attempts),
            ).fetchone()
        return row["id"] if row else None

    def claim(self) -> Job | None:
        """Take the next due job, or None. FOR UPDATE SKIP LOCKED is the key: a row another
        worker has already grabbed is skipped, not waited for, so two workers never get
        the same job and neither blocks the other."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "UPDATE ingest_jobs SET status = 'running', attempts = attempts + 1, locked_at = now() "
                "WHERE id = ("
                "  SELECT id FROM ingest_jobs WHERE status = 'queued' AND run_after <= now() "
                "  ORDER BY run_after, id FOR UPDATE SKIP LOCKED LIMIT 1) "
                "RETURNING id, document_id, attempts, max_attempts"
            ).fetchone()
        if row is None:
            return None
        return Job(row["id"], str(row["document_id"]), row["attempts"], row["max_attempts"])

    def complete(self, job_id: int) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE ingest_jobs SET status = 'done', finished_at = now(), locked_at = NULL "
                "WHERE id = %s",
                (job_id,),
            )

    def fail_or_retry(self, job: Job, error: str, backoff_seconds: float) -> bool:
        """After a failed attempt: queue it again after a delay, or give up when the attempts
        are used up. Returns True when it gave up."""
        gave_up = job.attempts >= job.max_attempts
        with self._pool.connection() as conn:
            if gave_up:
                conn.execute(
                    "UPDATE ingest_jobs SET status = 'failed', finished_at = now(), locked_at = NULL, "
                    "last_error = %s WHERE id = %s",
                    (error, job.id),
                )
            else:
                conn.execute(
                    "UPDATE ingest_jobs SET status = 'queued', locked_at = NULL, last_error = %s, "
                    "run_after = now() + make_interval(secs => %s) WHERE id = %s",
                    (error, float(backoff_seconds), job.id),
                )
        return gave_up

    def recover_stale(self, stale_after_seconds: float) -> list[str]:
        """Jobs stuck "running" (the worker died) go back in the queue. Ones that already used
        all their attempts are failed instead; their document ids are returned."""
        stale = "status = 'running' AND locked_at < now() - make_interval(secs => %s)"
        age = float(stale_after_seconds)
        with self._pool.connection() as conn:
            gave_up = conn.execute(
                "UPDATE ingest_jobs SET status = 'failed', finished_at = now(), locked_at = NULL, "
                "last_error = 'WorkerStopped' "
                f"WHERE {stale} AND attempts >= max_attempts RETURNING document_id",
                (age,),
            ).fetchall()
            conn.execute(
                "UPDATE ingest_jobs SET status = 'queued', run_after = now(), locked_at = NULL "
                f"WHERE {stale} AND attempts < max_attempts",
                (age,),
            )
        return [str(row["document_id"]) for row in gave_up]

    def enqueue_orphans(self, min_age_seconds: float, max_attempts: int = 3) -> list[str]:
        """Documents still "processing" that have never had a job (the app crashed between
        creating the document and queueing it). Old enough ones get a job."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "INSERT INTO ingest_jobs (document_id, max_attempts) "
                "SELECT d.id, %s FROM documents d "
                "WHERE d.status = 'processing' AND d.created_at < now() - make_interval(secs => %s) "
                "AND NOT EXISTS (SELECT 1 FROM ingest_jobs j WHERE j.document_id = d.id) "
                "RETURNING document_id",
                (max_attempts, float(min_age_seconds)),
            ).fetchall()
        return [str(row["document_id"]) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        """How many jobs are in each state (for the readiness check and the metrics)."""
        counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
        with self._pool.connection() as conn:
            for row in conn.execute("SELECT status, count(*) AS n FROM ingest_jobs GROUP BY status").fetchall():
                counts[row["status"]] = row["n"]
        return counts

    def latest_for_document(self, document_id: str) -> dict | None:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT id, status, attempts, max_attempts, last_error FROM ingest_jobs "
                "WHERE document_id = %s ORDER BY id DESC LIMIT 1",
                (document_id,),
            ).fetchone()
