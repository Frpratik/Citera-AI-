"""Pydantic models for the chat endpoints."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

AnswerType = Literal[
    "document_grounded",  # answer is supported by cited document passages
    "general",  # general conversation / general knowledge, NOT from the documents
    "insufficient_evidence",  # documents do not contain enough information
    "conflicting_information",  # documents disagree with each other
]

EvidenceStatus = Literal["strong", "moderate", "none", "not_applicable"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class SourceOut(BaseModel):
    document_id: str
    document_name: str
    page: int | None = None
    chunk_id: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    answer_type: AnswerType
    sources: list[SourceOut] = []
    # Based on retrieval similarity scores, NOT on the model's own opinion of itself.
    evidence_status: EvidenceStatus
    route: str  # how the question was classified (useful for debugging/learning)
    warnings: list[str] = []


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    answer_type: str | None = None
    created_at: str


class HistoryOut(BaseModel):
    messages: list[HistoryMessage]


class ClearHistoryOut(BaseModel):
    cleared: int
