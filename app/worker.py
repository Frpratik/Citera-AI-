"""Run a background worker as its own process:   python -m app.worker

Use this in production (with EMBEDDED_WORKER=false for the web app), so heavy PDF
processing never competes with web requests, and so you can run as many workers as you need.
"""

import logging
import signal
import threading

from app.api.deps import build_services
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.migrate import run_migrations
from app.db.pool import create_pool
from app.jobs.worker import WorkerLoop

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    run_migrations(settings.database_url)
    pool = create_pool(settings.database_url)
    services = build_services(settings, pool)

    loop = WorkerLoop(services.worker, settings)
    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM"):  # Ctrl+C, and the stop signal servers send
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stop.set())

    loop.start()
    try:
        while not stop.wait(1):  # short waits, so Ctrl+C is noticed quickly on Windows too
            pass
    finally:
        logger.info("Shutting down the worker")
        loop.stop()  # lets the current job finish
        pool.close()


if __name__ == "__main__":
    main()
