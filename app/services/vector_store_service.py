"""FAISS vector stores: one small index per document.

Why one index per document instead of one shared index?
FAISS indexes are awkward to delete from (many index types cannot remove vectors,
and rebuilding everything after each delete is wasteful). With one index per
document:

  * upload  -> build a new index folder
  * delete  -> remove that folder, so its chunks can never be retrieved again
  * isolation -> folders live under data/indexes/<owner_id>/<document_id>/
  * chatting -> never rebuilds anything; we only load indexes and search

Trade-off: a query searches each of the user's documents. That is fine for tens of
documents; at larger scale you would move to a vector database with metadata
filtering (e.g. pgvector, Qdrant).

Security note: FAISS "load_local" uses pickle for the docstore. We only load files
this application wrote itself, so `allow_dangerous_deserialization=True` is
acceptable here. Never load index files from untrusted sources.
"""

import logging
import shutil
import threading
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from app.services.embedding_service import EmbeddingService
from app.services.ids import check_ids, safe_child_path

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self, indexes_dir: Path, embedding_service: EmbeddingService):
        self._indexes_dir = indexes_dir
        self._embedding_service = embedding_service
        self._cache: dict[tuple[str, str], FAISS] = {}
        self._lock = threading.Lock()

    def _index_path(self, owner_id: str, document_id: str) -> Path:
        check_ids(owner_id, document_id)
        return safe_child_path(self._indexes_dir, owner_id, document_id)

    def index_exists(self, owner_id: str, document_id: str) -> bool:
        return (self._index_path(owner_id, document_id) / "index.faiss").exists()

    def build_index(self, owner_id: str, document_id: str, chunks: list[Document]) -> None:
        """Embed the chunks, build a FAISS index and persist it atomically."""
        final_path = self._index_path(owner_id, document_id)
        building_path = final_path.with_name(final_path.name + ".building")

        store = FAISS.from_documents(
            chunks,
            self._embedding_service.embeddings,
            # Inner product on normalised vectors = cosine similarity (higher = more similar).
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # Write to a temporary folder first so a crash never leaves a half-written index.
        shutil.rmtree(building_path, ignore_errors=True)
        building_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(str(building_path))
        shutil.rmtree(final_path, ignore_errors=True)
        building_path.rename(final_path)

        with self._lock:
            self._cache[(owner_id, document_id)] = store
        logger.info("Built index for document %s (%d chunks)", document_id, len(chunks))

    def _load(self, owner_id: str, document_id: str) -> FAISS:
        key = (owner_id, document_id)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        path = self._index_path(owner_id, document_id)
        store = FAISS.load_local(
            str(path),
            self._embedding_service.embeddings,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            allow_dangerous_deserialization=True,  # our own files only, see module docstring
        )
        with self._lock:
            self._cache[key] = store
        return store

    def search(
        self, owner_id: str, document_id: str, query_vector: list[float], k: int
    ) -> list[tuple[Document, float]]:
        """Return up to `k` (chunk, cosine_score) pairs from one document's index."""
        store = self._load(owner_id, document_id)
        results = store.similarity_search_with_score_by_vector(query_vector, k=k)
        return [(doc, float(score)) for doc, score in results]

    def delete_index(self, owner_id: str, document_id: str) -> None:
        with self._lock:
            self._cache.pop((owner_id, document_id), None)
        path = self._index_path(owner_id, document_id)
        shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(path.with_name(path.name + ".building"), ignore_errors=True)
