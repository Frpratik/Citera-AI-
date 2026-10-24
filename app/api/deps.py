"""Shared FastAPI dependencies: service wiring, client identification, rate limiting.

IMPORTANT about "users": this learning version has no login. The browser creates a
random client id and sends it in the `X-Client-ID` header; every document, index and
chat message is scoped to it. That prevents accidental mixing, but it is NOT security:
anyone who knows or guesses an id can use it. A real deployment needs authentication
(e.g. OAuth/JWT), authorization and tenant isolation per organization or user.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Depends, Header

from app.core.config import Settings, get_settings
from app.core.errors import MissingClientIdError, RateLimitExceededError
from app.graph.workflow import build_workflow
from app.repositories.document_repository import DocumentRepository
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.ids import OWNER_ID_PATTERN
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService


class RateLimiter:
    """Very small in-memory sliding-window limiter (per process, per client id)."""

    def __init__(self, limit_per_minute: int):
        self._limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self._limit:
                raise RateLimitExceededError("Too many requests. Please wait a moment and try again.")
            hits.append(now)


@dataclass
class Services:
    settings: Settings
    repository: DocumentRepository
    vector_store: VectorStoreService
    retrieval: RetrievalService
    documents: DocumentService
    chat: ChatService
    llm: LLMService
    rate_limiter: RateLimiter


def build_services(settings: Settings, embeddings=None, llm: LLMService | None = None) -> Services:
    """Create every service and connect them. Tests pass fakes for `embeddings` and `llm`."""
    settings.ensure_directories()

    repository = DocumentRepository(settings.db_path)
    embedding_service = EmbeddingService(settings.embedding_model, embeddings)
    vector_store = VectorStoreService(settings.indexes_dir, embedding_service)
    pdf_service = PDFService(
        uploads_dir=settings.uploads_dir,
        max_upload_bytes=settings.max_upload_bytes,
        max_pages=settings.max_pdf_pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retrieval = RetrievalService(
        repository=repository,
        vector_store=vector_store,
        embedding_service=embedding_service,
        top_k=settings.retrieval_top_k,
        min_score=settings.min_relevance_score,
    )
    llm_service = llm or LLMService(settings)
    workflow = build_workflow(llm_service, retrieval, settings)

    return Services(
        settings=settings,
        repository=repository,
        vector_store=vector_store,
        retrieval=retrieval,
        documents=DocumentService(
            repository, pdf_service, vector_store, settings.max_documents_per_user
        ),
        chat=ChatService(workflow, repository, settings.history_turns),
        llm=llm_service,
        rate_limiter=RateLimiter(settings.rate_limit_per_minute),
    )


_services: Services | None = None
_services_lock = threading.Lock()


def get_services() -> Services:
    """FastAPI dependency returning the application's services (created once)."""
    global _services
    if _services is None:
        with _services_lock:
            if _services is None:
                _services = build_services(get_settings())
    return _services


def get_client_id(x_client_id: str | None = Header(default=None)) -> str:
    if not x_client_id or not OWNER_ID_PATTERN.match(x_client_id):
        raise MissingClientIdError(
            "Missing or invalid X-Client-ID header (8-64 letters, digits, '-' or '_')."
        )
    return x_client_id


def enforce_rate_limit(
    client_id: str = Depends(get_client_id), services: Services = Depends(get_services)
) -> str:
    """Dependency for expensive endpoints. Returns the client id after the check passes."""
    services.rate_limiter.check(client_id)
    return client_id
