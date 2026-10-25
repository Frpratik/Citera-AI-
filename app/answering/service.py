import logging

from app.answering.checks import (
    build_sources,
    evidence_status,
    find_unverified_figures,
    valid_source_ids,
)
from app.answering.generator import AnswerGenerator
from app.answering.schemas import AnswerResult
from app.core.errors import LLMUnavailableError
from app.repositories.chunks import SearchHit

logger = logging.getLogger(__name__)

NO_EVIDENCE_MESSAGE = (
    "I couldn't find information about that in your uploaded documents, so I can't answer it "
    "reliably. Try rephrasing the question, or upload a document that covers it."
)
UNVERIFIED_CITATION_MESSAGE = (
    "I found related passages, but I couldn't confirm that they answer your question, so I'd "
    "rather not guess. Please check the listed documents directly or rephrase the question."
)


PASSAGES_SHOWN = 3
PASSAGES_ONLY_MESSAGE = (
    "The AI that writes answers is temporarily unavailable, so here are the most relevant "
    "passages from your documents instead."
)


class AnswerService:
    """Chunks in, a CHECKED answer out. We never pass the AI's claims on unchecked."""

    def __init__(self, generator: AnswerGenerator, strong_score: float = 0.8):
        self._generator = generator
        self._strong_score = strong_score

    def answer(self, question: str, chunks: list[SearchHit]) -> AnswerResult:
        # No evidence at all: never call the AI. It cannot make things up if it is never asked.
        if not chunks:
            return self._insufficient(NO_EVIDENCE_MESSAGE)

        try:
            draft = self._generator.generate(question, chunks)
        except LLMUnavailableError:
            # The AI is down, but search worked: show the best passages instead of an error.
            logger.warning("AI unavailable: returning the matching passages only")
            return self._passages_only(chunks)

        if not draft.sufficient_evidence:
            return self._insufficient(draft.answer.strip())  # the AI explains what is missing

        ids = valid_source_ids(draft.used_source_ids, len(chunks))
        if not ids:
            # It claims support but cites nothing we can verify: do not present it as grounded.
            logger.warning("Answer rejected: no valid citations")
            return self._insufficient(UNVERIFIED_CITATION_MESSAGE)

        cited = [chunks[i - 1] for i in ids]
        warnings: list[str] = []
        unverified = find_unverified_figures(draft.answer, [hit.text for hit in cited])
        if unverified:
            warnings.append(
                "These figures could not be found in the cited text, so please double-check "
                "them in the source: " + ", ".join(unverified)
            )

        return AnswerResult(
            answer=draft.answer.strip(),
            answer_type="conflicting_information" if draft.conflict_detected else "document_grounded",
            sources=build_sources(cited),
            evidence_status=evidence_status(cited, self._strong_score),
            warnings=warnings,
        )

    def _passages_only(self, chunks: list[SearchHit]) -> AnswerResult:
        """A useful answer without the AI: the best-matching passages, each with its place."""
        top = chunks[:PASSAGES_SHOWN]
        lines = [
            f"[{number}] {hit.document_name}, page {hit.page}: {hit.text[:300].strip()}"
            for number, hit in enumerate(top, start=1)
        ]
        return AnswerResult(
            answer=PASSAGES_ONLY_MESSAGE + "\n\n" + "\n\n".join(lines),
            answer_type="passages_only",
            sources=build_sources(top),
            evidence_status=evidence_status(top, self._strong_score),
        )

    @staticmethod
    def _insufficient(message: str) -> AnswerResult:
        return AnswerResult(
            answer=message, answer_type="insufficient_evidence", sources=[], evidence_status="none"
        )
