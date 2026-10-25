"""The measuring tools. Small, pure functions, so they are easy to test and to trust."""

import re
from collections.abc import Iterable

from app.repositories.chunks import SearchHit


def first_relevant_rank(hits: list[SearchHit], relevant: Iterable[tuple[str, int]]) -> int | None:
    """Position (1 = top) of the first result from a place that really holds the answer."""
    wanted = set(relevant)
    for rank, hit in enumerate(hits, start=1):
        if (hit.document_name, hit.page) in wanted:
            return rank
    return None


def hit_at(rank: int | None, k: int) -> bool:
    """Was a relevant result among the top k?"""
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    """1 for first place, 1/2 for second, 1/3 for third... 0 if never found.
    Averaged over questions this is MRR: how near the top the right chunk usually is."""
    return 1.0 / rank if rank else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _normalise(text: str) -> str:
    text = text.lower().replace("‑", "-").replace("–", "-").replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 85,000 -> 85000
    return re.sub(r"\s+", " ", text)


def facts_present(answer: str, groups: Iterable[Iterable[str]]) -> bool:
    """True if, for EVERY group, at least one alternative appears in the answer as a whole
    word or number (so "6" does not match inside "16")."""
    text = _normalise(answer)
    for group in groups:
        if not any(
            re.search(r"(?<![\w.])" + re.escape(_normalise(option)) + r"(?![\w])", text)
            for option in group
        ):
            return False
    return True


def contains_any(answer: str, fragments: Iterable[str]) -> bool:
    text = _normalise(answer)
    return any(_normalise(f) in text for f in fragments)
