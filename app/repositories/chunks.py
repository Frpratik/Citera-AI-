from dataclasses import dataclass

import numpy as np
from psycopg_pool import ConnectionPool

from app.ingestion.models import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str
    score: float  # meaning depends on who set it: cosine, keyword rank, or reranker score


def _to_hit(row: dict) -> SearchHit:
    return SearchHit(
        chunk_id=row["id"],
        document_id=str(row["document_id"]),
        document_name=row["filename"],
        page=row["page"],
        text=row["content"],
        score=float(row["score"]),
    )


class ChunkRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def add(self, document_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert all chunks of a document in ONE transaction: all stored or none."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        rows = [
            (c.chunk_id, document_id, c.page, c.index, c.text, np.asarray(v, dtype=np.float32))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, document_id, page, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                rows,
            )

    def delete_for_document(self, document_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

    def vector_search(
        self, owner_id: str, query_vector: list[float], top_k: int
    ) -> list[SearchHit]:
        """Meaning search: nearest chunks (cosine) across the owner's READY documents.

        `<=>` is pgvector's cosine DISTANCE (0 = identical), so similarity = 1 - distance.
        The owner filter is part of the query itself: tenant isolation is enforced by the
        database, not by hoping the application remembers to filter afterwards.
        """
        vector = np.asarray(query_vector, dtype=np.float32)
        with self._pool.connection() as conn, conn.transaction():
            # With a WHERE filter, a plain HNSW scan can return fewer than top_k rows
            # (it filters AFTER finding its candidates). Iterative scan keeps searching.
            conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            rows = conn.execute(
                "SELECT c.id, c.document_id, d.filename, c.page, c.content, "
                "       1 - (c.embedding <=> %(v)s) AS score "
                "FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE d.owner_id = %(owner)s AND d.status = 'ready' "
                "ORDER BY c.embedding <=> %(v)s LIMIT %(k)s",
                {"v": vector, "owner": owner_id, "k": top_k},
            ).fetchall()
        return [_to_hit(r) for r in rows]

    def keyword_search(self, owner_id: str, text: str, top_k: int) -> list[SearchHit]:
        """Keyword search: chunks containing ANY of the question's words, best match first.

        plainto_tsquery cleans the question the same way the chunks were cleaned
        (lowercase, stemming, stop words dropped) and joins the words with AND. We swap
        AND for OR, otherwise one word missing from a chunk ("say" in "What does clause
        7.3 say?") would hide an otherwise perfect match. ts_rank puts chunks that match
        more of the words first.
        """
        with self._pool.connection() as conn:
            rows = conn.execute(
                "WITH q AS (SELECT replace(plainto_tsquery('english', %(text)s)::text, '&', '|')"
                "           ::tsquery AS tsq) "
                "SELECT c.id, c.document_id, d.filename, c.page, c.content, "
                "       ts_rank(c.content_tsv, q.tsq) AS score "
                "FROM chunks c JOIN documents d ON d.id = c.document_id CROSS JOIN q "
                "WHERE d.owner_id = %(owner)s AND d.status = 'ready' AND c.content_tsv @@ q.tsq "
                "ORDER BY score DESC LIMIT %(k)s",
                {"text": text, "owner": owner_id, "k": top_k},
            ).fetchall()
        return [_to_hit(r) for r in rows]
