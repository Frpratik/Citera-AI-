from concurrent.futures import ThreadPoolExecutor

from app.repositories.documents import DocumentRepository
from app.repositories.jobs import JobRepository


def make_document(db_pool, name="a.pdf") -> str:
    return DocumentRepository(db_pool).create("alice-owner", name).id


def run_sql(db_pool, sql, params=()):
    with db_pool.connection() as conn:
        conn.execute(sql, params)


def status_of(db_pool, job_id) -> str:
    with db_pool.connection() as conn:
        return conn.execute("SELECT status FROM ingest_jobs WHERE id = %s", (job_id,)).fetchone()["status"]


def test_claim_hands_out_a_job_once_and_counts_the_attempt(db_pool):
    jobs = JobRepository(db_pool)
    doc_id = make_document(db_pool)
    job_id = jobs.enqueue(doc_id, max_attempts=3)

    job = jobs.claim()
    assert (job.id, job.document_id, job.attempts, job.max_attempts) == (job_id, doc_id, 1, 3)
    assert status_of(db_pool, job_id) == "running"
    assert jobs.claim() is None  # nothing else is waiting


def test_a_document_cannot_have_two_active_jobs(db_pool):
    jobs = JobRepository(db_pool)
    doc_id = make_document(db_pool)
    first = jobs.enqueue(doc_id)
    assert first is not None and jobs.enqueue(doc_id) is None  # refused by the database

    jobs.complete(jobs.claim().id)
    assert jobs.enqueue(doc_id) is not None  # fine again once the first is finished


def test_a_job_scheduled_for_later_is_not_claimed_yet(db_pool):
    jobs = JobRepository(db_pool)
    job_id = jobs.enqueue(make_document(db_pool))
    run_sql(db_pool, "UPDATE ingest_jobs SET run_after = now() + interval '1 hour' WHERE id = %s", (job_id,))
    assert jobs.claim() is None


def test_several_workers_claiming_at_once_never_get_the_same_job(db_pool):
    jobs = JobRepository(db_pool)
    queued = {jobs.enqueue(make_document(db_pool, f"{i}.pdf")) for i in range(4)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda _: jobs.claim(), range(4)))

    ids = [job.id for job in claimed if job is not None]
    assert len(ids) == 4 and set(ids) == queued  # four jobs, four different claims


def test_complete_marks_the_job_done(db_pool):
    jobs = JobRepository(db_pool)
    job_id = jobs.enqueue(make_document(db_pool))
    jobs.complete(jobs.claim().id)
    assert status_of(db_pool, job_id) == "done"


def test_a_failed_attempt_is_retried_after_the_delay(db_pool):
    jobs = JobRepository(db_pool)
    job_id = jobs.enqueue(make_document(db_pool), max_attempts=3)

    assert jobs.fail_or_retry(jobs.claim(), "RuntimeError", backoff_seconds=3600) is False
    assert status_of(db_pool, job_id) == "queued"
    assert jobs.claim() is None  # the hour has not passed

    run_sql(db_pool, "UPDATE ingest_jobs SET run_after = now() WHERE id = %s", (job_id,))
    assert jobs.claim().attempts == 2


def test_the_job_gives_up_when_the_attempts_are_used_up(db_pool):
    jobs = JobRepository(db_pool)
    doc_id = make_document(db_pool)
    job_id = jobs.enqueue(doc_id, max_attempts=2)

    assert jobs.fail_or_retry(jobs.claim(), "RuntimeError", backoff_seconds=0) is False
    assert jobs.fail_or_retry(jobs.claim(), "RuntimeError", backoff_seconds=0) is True
    assert status_of(db_pool, job_id) == "failed"
    assert jobs.latest_for_document(doc_id)["last_error"] == "RuntimeError"


def test_stale_running_jobs_are_requeued_or_failed(db_pool):
    jobs = JobRepository(db_pool)
    retryable = jobs.enqueue(make_document(db_pool, "a.pdf"), max_attempts=3)
    exhausted_doc = make_document(db_pool, "b.pdf")
    exhausted = jobs.enqueue(exhausted_doc, max_attempts=1)
    fresh = jobs.enqueue(make_document(db_pool, "c.pdf"))
    for _ in range(3):
        jobs.claim()  # all three are now "running", with attempts = 1
    run_sql(
        db_pool,
        "UPDATE ingest_jobs SET locked_at = now() - interval '1 hour' WHERE id = ANY(%s)",
        ([retryable, exhausted],),
    )

    gave_up = jobs.recover_stale(stale_after_seconds=900)

    assert gave_up == [exhausted_doc]
    assert status_of(db_pool, retryable) == "queued"  # a try left: back in the queue
    assert status_of(db_pool, exhausted) == "failed"  # no tries left
    assert status_of(db_pool, fresh) == "running"  # recently started: left alone


def test_documents_stuck_without_any_job_get_one_but_only_when_old_enough(db_pool):
    jobs = JobRepository(db_pool)
    old = make_document(db_pool, "old.pdf")
    make_document(db_pool, "new.pdf")
    has_job = make_document(db_pool, "has-job.pdf")
    jobs.enqueue(has_job)
    run_sql(db_pool, "UPDATE documents SET created_at = now() - interval '1 hour' WHERE id IN (%s, %s)", (old, has_job))

    assert jobs.enqueue_orphans(min_age_seconds=120) == [old]
    assert jobs.enqueue_orphans(min_age_seconds=120) == []  # now it has a job; no repeats
