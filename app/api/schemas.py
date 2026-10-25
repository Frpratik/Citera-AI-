import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class LoginRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL.match(value):
            raise ValueError("not a valid email address")
        return value


class RegisterRequest(LoginRequest):
    password: str = Field(min_length=8, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    created_at: datetime


class DocumentOut(BaseModel):
    """One document as the API shows it (the owner is never exposed)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: Literal["processing", "ready", "failed"]
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    created_at: datetime


class DocumentListOut(BaseModel):
    documents: list[DocumentOut]


class DeleteOut(BaseModel):
    id: str
    deleted: bool


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class HistoryMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    role: Literal["user", "assistant"]
    content: str
    answer_type: str | None = None
    created_at: datetime


class HistoryOut(BaseModel):
    messages: list[HistoryMessage]


class ClearHistoryOut(BaseModel):
    cleared: int


class UsageOut(BaseModel):
    """What the signed-in user's questions used over the period."""

    hours: int
    questions: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    avg_response_ms: int
    fallbacks: int  # questions that needed the fallback model
    estimated_cost: float | None  # only when prices are set in the settings
