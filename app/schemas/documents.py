"""Pydantic models for the document endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

DocumentStatus = Literal["processing", "ready", "failed"]


class DocumentOut(BaseModel):
    """One uploaded document as the API returns it."""

    model_config = ConfigDict(from_attributes=True)  # lets us build it from a DocumentRecord

    id: str
    filename: str
    status: DocumentStatus
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    created_at: str


class DocumentListOut(BaseModel):
    documents: list[DocumentOut]


class DocumentStatusOut(BaseModel):
    id: str
    status: DocumentStatus
    error: str | None = None


class DeleteOut(BaseModel):
    id: str
    deleted: bool
