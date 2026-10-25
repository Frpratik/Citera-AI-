import io
import time
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.errors import NotFoundError, ValidationFailedError
from app.jobs.worker import Worker, WorkerLoop
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.jobs import JobRepository
from app.services.ingestion_service import MISSING_FILE_MESSAGE, IngestionService
from tests.fakes import FakeEmbedder, FlakyEmbedder
from tests.pdf_factory import make_pdf

TEXT = "Refund requests must be filed within 14 days of purchase."


@pytest.fixture
def make_parts(db_pool, tmp_path):
    """Builds the ingestion service and a worker. Retries have no delay unless asked."""

    def make(embedder=None, **overrides):
        options = {"job_retry_backoff_seconds": 0.0, "job_max_attempts": 3} | overrides
        settings = Settings(data_dir=tmp_path, _env_file=None, **options)
        jobs = JobRepository(db_pool)
        ingestion = IngestionService(
            DocumentRepository(db_pool), ChunkRepository(db_pool), jobs, embedder or FakeEmbedder(), settings
        )
        return SimpleNamespace(
            ingestion=ingestion, worker=Worker(jobs, ingestion, settings), jobs=jobs, settings=settings,
            docs=DocumentRepository(db_pool), chunks=ChunkRepository(db_pool), pool=db_pool, root=tmp_path,
        )

    return make


def accept(parts, pages=(TEXT,), owner="alice-owner"):
    return parts.ingestion.accept(owner, "terms.pdf", io.BytesIO(make_pdf(list(pages))))


def document(parts, doc_id, owner="alice-owner"):
    return parts.docs.get(owner, doc_id)


def job(parts, doc_id):
    return parts.jobs.latest_for_document(doc_id)


def test_the_happy_path(make_parts):
    parts = make_parts()
    doc = accept(parts)
    assert document(parts, doc.id).status == "processing"

    assert parts.worker.run_once() is True
    ready = document(parts, doc.id)
    assert ready.status == "ready" and ready.chunk_count == 1
    assert job(parts, doc.id)["status"] == "done"
    assert parts.worker.run_once() is False  # queue is empty


def test_a_broken_file_fails_at_once_without_pointless_retries(make_parts):
    parts = make_parts()
    doc = accept(parts, pages=("", ""))  # a PDF with no text

    parts.worker.drain()

    failed = document(parts, doc.id)
    assert failed.status == "failed" and "No text" in failed.error
    assert job(parts, doc.id)["attempts"] == 1 and job(parts, doc.id)["status"] == "done"


def test_a_temporary_problem_is_retried_and_then_succeeds(make_parts):
    parts = make_parts(embedder=FlakyEmbedder(failures=1))
    doc = accept(parts)

    parts.worker.run_once()  # the embedding model "hiccups"
    assert document(parts, doc.id).status == "processing"
    assert job(parts, doc.id)["status"] == "queued" and job(parts, doc.id)["last_error"] == "RuntimeError"

    parts.worker.run_once()  # second attempt works
    assert document(parts, doc.id).status == "ready"
    assert job(parts, doc.id)["attempts"] == 2 and job(parts, doc.id)["status"] == "done"


def test_a_problem_that_never_goes_away_ends_as_failed_with_a_message(make_parts):
    parts = make_parts(embedder=FlakyEmbedder(failures=99))
    doc = accept(parts)

    parts.worker.drain()

    failed = document(parts, doc.id)
    assert failed.status == "failed" and "after 3 attempts" in failed.error
    assert "hiccup" not in failed.error  # internals are not shown to the user
    assert job(parts, doc.id)["status"] == "failed" and job(parts, doc.id)["attempts"] == 3
    assert parts.chunks.vector_search("alice-owner", [1.0] + [0.0] * 383, 5) == []


def test_a_retry_waits_for_its_delay(make_parts):
    parts = make_parts(embedder=FlakyEmbedder(failures=1), job_retry_backoff_seconds=3600.0)
    accept(parts)
    assert parts.worker.run_once() is True  # first attempt fails
    assert parts.worker.run_once() is False  # the retry is an hour away


def test_processing_twice_does_not_duplicate_chunks(make_parts):
    parts = make_parts()
    doc = accept(parts)
    parts.ingestion.process(doc.id)
    parts.ingestion.process(doc.id)  # e.g. a retry after a crash that came too late
    assert document(parts, doc.id).chunk_count == 1
    with parts.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 1


def test_a_document_deleted_while_waiting_is_simply_skipped(make_parts):
    parts = make_parts()
    doc = accept(parts)
    parts.ingestion.delete("alice-owner", doc.id)  # the job goes with it
    assert parts.worker.run_once() is False
    parts.ingestion.process(doc.id)  # and calling process directly is harmless too


def test_a_missing_file_fails_with_a_clear_message(make_parts):
    parts = make_parts()
    doc = accept(parts)
    (parts.root / "uploads" / f"{doc.id}.pdf").unlink()
    parts.worker.drain()
    assert document(parts, doc.id).error == MISSING_FILE_MESSAGE


def test_reprocessing_a_failed_document_works_once_the_problem_is_fixed(make_parts):
    embedder = FlakyEmbedder(failures=99)
    parts = make_parts(embedder=embedder)
    doc = accept(parts)
    parts.worker.drain()
    assert document(parts, doc.id).status == "failed"

    embedder.failures = 0  # the problem is fixed
    again = parts.ingestion.reprocess("alice-owner", doc.id)
    assert again.status == "processing" and again.error is None
    parts.worker.drain()
    assert document(parts, doc.id).status == "ready"


def test_reprocess_rules(make_parts):
    parts = make_parts()
    ready = accept(parts)
    parts.worker.drain()
    with pytest.raises(ValidationFailedError):
        parts.ingestion.reprocess("alice-owner", ready.id)  # only failed ones
    with pytest.raises(NotFoundError):
        parts.ingestion.reprocess("bob-owner", ready.id)  # not his

    failed = accept(parts, pages=("", ""))
    parts.worker.drain()
    (parts.root / "uploads" / f"{failed.id}.pdf").unlink()
    assert parts.ingestion.reprocess("alice-owner", failed.id).error == MISSING_FILE_MESSAGE


def test_recover_requeues_dead_jobs_and_fails_the_exhausted_ones(make_parts):
    parts = make_parts()
    retryable, exhausted = accept(parts), accept(parts)
    with parts.pool.connection() as conn:
        conn.execute("UPDATE ingest_jobs SET max_attempts = 1 WHERE document_id = %s", (exhausted.id,))
    parts.jobs.claim()
    parts.jobs.claim()  # both look like a worker took them and then died
    with parts.pool.connection() as conn:
        conn.execute("UPDATE ingest_jobs SET locked_at = now() - interval '1 hour'")

    parts.worker.recover()

    assert job(parts, retryable.id)["status"] == "queued"
    assert job(parts, exhausted.id)["status"] == "failed"
    assert document(parts, exhausted.id).status == "failed"
    parts.worker.drain()
    assert document(parts, retryable.id).status == "ready"


def test_recover_gives_a_job_to_a_document_that_lost_its_job_in_a_crash(make_parts):
    parts = make_parts()
    doc = accept(parts)
    with parts.pool.connection() as conn:
        conn.execute("DELETE FROM ingest_jobs")  # the crash: document saved, job never queued
        conn.execute("UPDATE documents SET created_at = now() - interval '1 hour'")

    parts.worker.recover()
    parts.worker.drain()

    assert document(parts, doc.id).status == "ready"


def test_a_brand_new_document_is_not_mistaken_for_a_lost_one(make_parts):
    parts = make_parts()
    parts.docs.create("alice-owner", "just-now.pdf")  # created a moment ago, no job yet
    parts.worker.recover()
    assert parts.jobs.claim() is None


def test_the_background_loop_processes_jobs_and_stops_cleanly(make_parts):
    parts = make_parts(worker_poll_seconds=0.05)
    loop = WorkerLoop(parts.worker, parts.settings)
    loop.start()
    try:
        doc = accept(parts)
        deadline = time.monotonic() + 10
        while document(parts, doc.id).status == "processing" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert document(parts, doc.id).status == "ready"
    finally:
        loop.stop(timeout=10)
    assert not loop._thread.is_alive()
