"""Validation of identifiers that end up inside filesystem paths.

Owner IDs come from the browser and document IDs from URLs, so they are untrusted.
We only ever build paths from IDs that match a strict pattern, and then double-check
that the final path is still inside the intended base folder.
"""

import re
from pathlib import Path

OWNER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def safe_child_path(base_dir: Path, *parts: str) -> Path:
    """Join `parts` onto `base_dir` and make sure the result cannot escape it."""
    base = base_dir.resolve()
    candidate = base.joinpath(*parts).resolve()
    if base != candidate and base not in candidate.parents:
        raise ValueError("Unsafe path")
    return candidate


def check_ids(owner_id: str, document_id: str | None = None) -> None:
    if not OWNER_ID_PATTERN.match(owner_id):
        raise ValueError("Invalid owner id")
    if document_id is not None and not DOCUMENT_ID_PATTERN.match(document_id):
        raise ValueError("Invalid document id")
