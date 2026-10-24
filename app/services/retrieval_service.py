"""Retrieval: find the chunks most relevant to a question across a user's documents."""

import logging
from dataclasses import dataclass, field

from app.core.errors import RetrievalUnavailableError
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    document_name: str
    page: int | None
    chunk_id: str
    text: str
    score: float  # cosine similarity, higher = more relevant


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    document_count: int = 0  # how many ready documents the user has


class RetrievalService:
    def __init__(
        self,
        repository: DocumentRepository,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        top_k: int,
        min_score: float,
    ):
        self._repository = repository
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, owner_id: str, question: str) -> RetrievalResult:
        # The SQLite lookup is what enforces ownership: we only ever search indexes
        # of documents that belong to this owner and are 'ready'.
        documents = self._repository.list_ready(owner_id)
        if not documents:
            return RetrievalResult(chunks=[], document_count=0)

        try:
            # Embed the question once, then reuse the vector for every document index.
            query_vector = self._embedding_service.embeddings.embed_query(question)

            candidates: list[RetrievedChunk] = []
            for document in documents:
                try:
                    hits = self._vector_store.search(owner_id, document.id, query_vector, self._top_k)
                except Exception:
                    # One broken index should not break chat for the other documents.
                    logger.exception("Search failed for document %s", document.id)
                    continue
                for chunk, score in hits:
                    candidates.append(
                        RetrievedChunk(
                            document_id=document.id,
                            document_name=document.filename,
                            page=chunk.metadata.get("page"),
                            chunk_id=chunk.metadata.get("chunk_id", ""),
                            text=chunk.page_content,
                            score=score,
                        )
                    )
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise RetrievalUnavailableError(
                "Document search is temporarily unavailable. Please try again."
            ) from exc

        candidates.sort(key=lambda item: item.score, reverse=True)
        relevant = [item for item in candidates[: self._top_k] if item.score >= self._min_score]
        return RetrievalResult(chunks=relevant, document_count=len(documents))
