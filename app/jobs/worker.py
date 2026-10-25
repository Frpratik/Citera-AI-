import logging
import threading
import time

from app.core.config import Settings
from app.repositories.jobs import JobRepository
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

GAVE_UP_MESSAGE = "Processing failed after {attempts} attempts. Try Reprocess, or upload the file again."


class Worker:
    """Takes jobs from the queue and runs them, one at a time."""

    def __init__(self, jobs: JobRepository, ingestion: IngestionService, settings: Settings):
        self._jobs = jobs
        self._ingestion = ingestion
        self._settings = settings

    def warm_up(self) -> None:
        """Load the embedding model before the first job, which would otherwise wait for it."""
        self._ingestion.warm_up()

    def run_once(self) -> bool:
        """Handle one job if one is due. Returns False when the queue had nothing for us."""
        job = self._jobs.claim()
        if job is None:
            return False
        logger.info("Job %d: document %s, attempt %d of %d", job.id, job.document_id, job.attempts, job.max_attempts)
        try:
            self._ingestion.process(job.document_id)
        except Exception as exc:
            logger.error("Job %d failed (%s)", job.id, type(exc).__name__, exc_info=exc)
            # Store only the error TYPE: messages can contain document text.
            backoff = self._settings.job_retry_backoff_seconds * job.attempts  # waits longer each time
            if self._jobs.fail_or_retry(job, type(exc).__name__, backoff):
                self._ingestion.mark_failed(job.document_id, GAVE_UP_MESSAGE.format(attempts=job.max_attempts))
        else:
            self._jobs.complete(job.id)
        return True

    def drain(self, max_jobs: int = 100) -> int:
        """Handle due jobs until none are left. Mostly for tests and scripts."""
        handled = 0
        while handled < max_jobs and self.run_once():
            handled += 1
        return handled

    def recover(self) -> None:
        """Repair what a crash can leave behind."""
        for document_id in self._jobs.recover_stale(self._settings.job_stale_after_seconds):
            self._ingestion.mark_failed(document_id, GAVE_UP_MESSAGE.format(attempts=self._settings.job_max_attempts))
        queued = self._jobs.enqueue_orphans(
            self._settings.job_orphan_min_age_seconds, self._settings.job_max_attempts
        )
        if queued:
            logger.warning("Queued %d document(s) that had no job", len(queued))


class WorkerLoop:
    """Runs a Worker on a background thread until stopped."""

    def __init__(self, worker: Worker, settings: Settings):
        self._worker = worker
        self._poll = settings.worker_poll_seconds
        self._recovery_every = settings.job_recovery_interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ingest-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()
        logger.info("Background worker started")

    def stop(self, timeout: float = 30.0) -> None:
        """Ask the loop to finish its current job and exit, then wait for it."""
        self._stop.set()
        self._thread.join(timeout)
        logger.info("Background worker stopped")

    def _run(self) -> None:
        try:
            self._worker.warm_up()
        except Exception:
            logger.exception("Warm-up failed")
        next_recovery = 0.0
        while not self._stop.is_set():
            handled = False
            try:
                if time.monotonic() >= next_recovery:
                    self._worker.recover()
                    next_recovery = time.monotonic() + self._recovery_every
                handled = self._worker.run_once()
            except Exception:
                logger.exception("Worker loop error")
            if not handled:
                self._stop.wait(self._poll)  # idle: sleep, but wake at once if asked to stop
