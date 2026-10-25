from typing import Literal

from pydantic import BaseModel, Field

AnswerType = Literal[
    "document_grounded",  # supported by cited chunks
    "insufficient_evidence",  # the documents do not contain enough to answer
    "conflicting_information",  # the documents disagree with each other
    "general",  # a general reply that did not use the documents at all
    "passages_only",  # the AI was unavailable: the best matching passages, no written answer
]

EvidenceStatus = Literal["strong", "moderate", "none", "not_applicable"]


class RagAnswer(BaseModel):
    """The form the AI must fill in. The field descriptions are shown to the AI."""

    answer: str = Field(description="The answer, written only from the provided sources.")
    sufficient_evidence: bool = Field(
        description="False if the sources do not contain enough information to answer."
    )
    used_source_ids: list[int] = Field(
        default_factory=list,
        description="Numbers of the sources that directly support the answer.",
    )
    conflict_detected: bool = Field(
        default=False, description="True if the sources contradict each other on the answer."
    )


class Source(BaseModel):
    """One citation shown to the user. Built from our own data, never from the AI's words."""

    document_id: str
    document_name: str
    page: int
    chunk_id: str
    snippet: str
    score: float


class AnswerResult(BaseModel):
    """What the rest of the app receives after the AI's answer has been checked."""

    answer: str
    answer_type: AnswerType
    sources: list[Source] = []
    evidence_status: EvidenceStatus
    warnings: list[str] = []


class ChatResult(AnswerResult):
    """An answer plus how the question was routed (useful for debugging and for the UI)."""

    route: str
