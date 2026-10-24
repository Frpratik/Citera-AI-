"""Graph nodes. Each node is a function: state in -> partial state update out."""

import logging
import re

from app.core.config import Settings
from app.graph.state import GraphState
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService, RetrievedChunk

logger = logging.getLogger(__name__)

NO_DOCUMENTS_MESSAGE = (
    "I can't answer that yet because you haven't uploaded any ready documents. "
    "Upload a PDF using the panel on the left and ask again."
)
NO_EVIDENCE_MESSAGE = (
    "I couldn't find information about that in your uploaded documents, so I can't answer it "
    "reliably. Try rephrasing the question, or upload a document that covers it."
)
UNVERIFIED_CITATION_MESSAGE = (
    "I found related passages, but I couldn't confirm that they answer your question, so I'd "
    "rather not guess. Please check the listed documents directly or rephrase the question."
)
FALLBACK_NOTE = "(I couldn't find this in your uploaded documents, so this is a general answer.)\n\n"

_CITATION_MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def find_unverified_figures(answer: str, cited_texts: list[str]) -> list[str]:
    """Return numbers in the answer that do not appear in the cited source text.

    A cheap evidence check: a wrong fee or percentage is the costliest hallucination,
    so every multi-digit number the model states must literally appear in the sources.
    (Single digits are ignored because they are usually counts or list numbering.)
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


class GraphNodes:
    """Holds the dependencies (LLM, retrieval) that the node functions need."""

    def __init__(self, llm: LLMService, retrieval: RetrievalService, settings: Settings):
        self._llm = llm
        self._retrieval = retrieval
        self._settings = settings

    # ------------------------------------------------------------------ classify
    def classify_question(self, state: GraphState) -> dict:
        decision = self._llm.classify(state["question"], state.get("history", []))
        logger.info("Question classified as %s", decision.route)
        return {"route": decision.route, "route_reason": decision.reason}

    # ------------------------------------------------------------------ retrieve
    def retrieve_documents(self, state: GraphState) -> dict:
        result = self._retrieval.retrieve(state["owner_id"], state["question"])
        logger.info(
            "Retrieved %d relevant chunks from %d documents",
            len(result.chunks),
            result.document_count,
        )
        return {"chunks": result.chunks, "document_count": result.document_count}

    # ------------------------------------------------------------------ grounded answer
    def generate_rag_answer(self, state: GraphState) -> dict:
        chunks: list[RetrievedChunk] = state.get("chunks", [])

        # No evidence at all: skip the LLM entirely. It cannot hallucinate if never asked.
        if not chunks:
            no_documents = state.get("document_count", 0) == 0
            return {
                "answer": NO_DOCUMENTS_MESSAGE if no_documents else NO_EVIDENCE_MESSAGE,
                "answer_type": "insufficient_evidence",
                "sources": [],
                "evidence_status": "none",
                "warnings": [],
            }

        result = self._llm.answer_rag(state["question"], chunks)
        return {
            "answer": result.answer.strip(),
            "sufficient_evidence": result.sufficient_evidence,
            "used_source_ids": result.used_source_ids,
            "conflict_detected": result.conflict_detected,
        }

    # ------------------------------------------------------------------ general answer
    def generate_general_answer(self, state: GraphState) -> dict:
        answer = self._llm.answer_general(state["question"], state.get("history", []))
        if state.get("route") == "uncertain":
            answer = FALLBACK_NOTE + answer
        return {
            "answer": answer.strip(),
            "answer_type": "general",
            "sources": [],
            "evidence_status": "not_applicable",
            "warnings": [],
        }

    # ------------------------------------------------------------------ validation
    def validate_answer(self, state: GraphState) -> dict:
        """Check the model's answer against what was actually retrieved.

        We do NOT trust the model's own claims. Sources shown to the user are built
        from real retrieval metadata, never from text the model wrote, and evidence
        strength comes from similarity scores, not from the model's self-assessment.
        """
        # The no-chunks case was already finalised by generate_rag_answer.
        if state.get("answer_type") == "insufficient_evidence":
            return {}

        chunks: list[RetrievedChunk] = state.get("chunks", [])

        if not state.get("sufficient_evidence", False):
            return {
                "answer": state["answer"],
                "answer_type": "insufficient_evidence",
                "sources": [],
                "evidence_status": "none",
                "warnings": [],
            }

        # Keep only citation numbers that point to a chunk we really retrieved.
        valid_ids: list[int] = []
        for source_id in state.get("used_source_ids", []):
            if 1 <= source_id <= len(chunks) and source_id not in valid_ids:
                valid_ids.append(source_id)

        if not valid_ids:
            # The model claims support but cites nothing verifiable: do not present it as grounded.
            return {
                "answer": UNVERIFIED_CITATION_MESSAGE,
                "answer_type": "insufficient_evidence",
                "sources": [],
                "evidence_status": "none",
                "warnings": [],
            }

        cited = [chunks[i - 1] for i in valid_ids]
        top_score = max(chunk.score for chunk in cited)
        evidence_status = "strong" if top_score >= self._settings.strong_relevance_score else "moderate"

        warnings: list[str] = []
        unverified = find_unverified_figures(state["answer"], [c.text for c in cited])
        if unverified:
            warnings.append(
                "These figures could not be found in the cited text, so please double-check them "
                "in the source: " + ", ".join(unverified)
            )

        # One entry per (document, page) so the list stays short and readable.
        sources: list[dict] = []
        seen: set[tuple[str, int | None]] = set()
        for chunk in cited:
            key = (chunk.document_id, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "snippet": chunk.text[:240].strip(),
                    "score": round(chunk.score, 3),
                }
            )

        answer_type = "conflicting_information" if state.get("conflict_detected") else "document_grounded"
        return {
            "answer": state["answer"],
            "answer_type": answer_type,
            "sources": sources,
            "evidence_status": evidence_status,
            "warnings": warnings,
        }
