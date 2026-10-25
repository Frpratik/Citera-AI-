import re

from app.answering.schemas import EvidenceStatus, Source
from app.repositories.chunks import SearchHit

_CITATION_MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")  # [1] or [1, 2]
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def valid_source_ids(used: list[int], chunk_count: int) -> list[int]:
    """Keep only citation numbers that point at a chunk we really sent, without repeats."""
    ids: list[int] = []
    for source_id in used:
        if 1 <= source_id <= chunk_count and source_id not in ids:
            ids.append(source_id)
    return ids


def find_unverified_figures(answer: str, cited_texts: list[str]) -> list[str]:
    """Numbers in the answer that do not appear in the cited text.

    A wrong fee or percentage is the costliest made-up detail, so every multi-digit
    number the AI states must literally appear in the sources it cited. Single digits
    are ignored (they are usually counts or list numbers), and so are "[1]" markers.
    """
    haystack = " ".join(cited_texts).replace(",", "")
    cleaned_answer = _CITATION_MARKER.sub("", answer)
    missing: list[str] = []
    for token in _NUMBER.findall(cleaned_answer):
        number = token.replace(",", "")
        if len(number.replace(".", "")) < 2:
            continue
        if number not in haystack and token not in missing:
            missing.append(token)
    return missing


def evidence_status(cited: list[SearchHit], strong_score: float) -> EvidenceStatus:
    """How strong the evidence is, from the re-ranker's scores (not the AI's own opinion)."""
    return "strong" if max(hit.score for hit in cited) >= strong_score else "moderate"


def build_sources(cited: list[SearchHit]) -> list[Source]:
    """One entry per (document, page), built from retrieval data, so the AI cannot invent one."""
    sources: list[Source] = []
    seen: set[tuple[str, int]] = set()
    for hit in cited:
        key = (hit.document_id, hit.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                document_id=hit.document_id,
                document_name=hit.document_name,
                page=hit.page,
                chunk_id=hit.chunk_id,
                snippet=hit.text[:240].strip(),
                score=round(hit.score, 3),
            )
        )
    return sources
