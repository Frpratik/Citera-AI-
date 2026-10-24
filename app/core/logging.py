"""Logging setup.

We log events (uploads, routing decisions, errors) but never API keys and never
document text or user questions, because those can be sensitive.
"""

import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    # Third-party libraries can be chatty at INFO level.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
