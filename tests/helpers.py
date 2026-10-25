from app.ingestion.models import Chunk
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository


def store_document(db_pool, embedder, owner, filename, texts, ready=True):
    """Put a document with one chunk per text straight into the database (no PDF needed)."""
    docs = DocumentRepository(db_pool)
    record = docs.create(owner, filename)
    chunks = [Chunk(f"{record.id}-c{i}", record.id, i + 1, i, t) for i, t in enumerate(texts)]
    ChunkRepository(db_pool).add(record.id, chunks, embedder.embed_documents(texts))
    if ready:
        docs.mark_ready(record.id, len(texts), len(texts))
    return record
