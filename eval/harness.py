"""The evaluation environment: a fresh throw-away database, the corpus ingested into it with
the REAL embedding and re-ranking models, and a way to run each search stage separately."""

import io
import tempfile
from dataclasses import replace
from pathlib import Path

import psycopg
from psycopg.conninfo import make_conninfo

from app.core.config import Settings, get_settings
from app.db.migrate import run_migrations
from app.db.pool import create_pool
from app.embeddings.service import SentenceTransformerEmbedder
from app.repositories.chunks import ChunkRepository, SearchHit
from app.repositories.documents import DocumentRepository
from app.repositories.jobs import JobRepository
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.services.ingestion_service import IngestionService
from eval.corpus import build_corpus
from tests.pdf_factory import make_pdf

EVAL_DB = "citera_eval"
OWNER = "eval-owner"
CANDIDATES = 20  # how many results each search keeps, as in the real pipeline

STAGES = ("dense", "keyword", "hybrid", "dense+rerank", "hybrid+rerank", "hybrid-wide+rerank")


class EvalEnvironment:
    def __init__(self):
        base = get_settings()
        self.base_settings = base
        with psycopg.connect(base.database_url, autocommit=True, connect_timeout=5) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {EVAL_DB} WITH (FORCE)")
            admin.execute(f"CREATE DATABASE {EVAL_DB}")
        self.url = make_conninfo(base.database_url, dbname=EVAL_DB)
        run_migrations(self.url)
        self.pool = create_pool(self.url, max_size=4)
        self.chunks = ChunkRepository(self.pool)
        self.documents = DocumentRepository(self.pool)
        self.embedder = SentenceTransformerEmbedder(base.embedding_model, base.embedding_batch_size)
        self.reranker = CrossEncoderReranker(base.rerank_model)
        self.embedder.warm_up()
        self.reranker.warm_up()
        self._tmp = Path(tempfile.mkdtemp(prefix="citera-eval-"))
        self.chunk_count = 0

    def load(
        self, chunk_size: int | None = None, chunk_overlap: int | None = None, haystack_pages: int = 6
    ) -> Settings:
        """(Re)build the index from the corpus with the given chunking. Returns the settings used."""
        base = self.base_settings
        settings = Settings(
            data_dir=self._tmp,
            chunk_size=chunk_size or base.chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else base.chunk_overlap,
            _env_file=None,
        )
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE documents CASCADE")
        ingestion = IngestionService(
            self.documents, self.chunks, JobRepository(self.pool), self.embedder, settings
        )
        for name, pages in build_corpus(haystack_pages=haystack_pages).items():
            record = ingestion.ingest_now(OWNER, name, io.BytesIO(make_pdf(pages)))
            if record.status != "ready":
                raise RuntimeError(f"{name} failed to load: {record.error}")
        with self.pool.connection() as conn:
            self.chunk_count = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        return settings

    def stages(self, query: str) -> dict[str, list[SearchHit]]:
        """The ranked results of every search stage for one query (best first)."""
        vector = self.embedder.embed_query(query)
        dense = self.chunks.vector_search(OWNER, vector, CANDIDATES)
        keyword = self.chunks.keyword_search(OWNER, query, CANDIDATES)
        merged = reciprocal_rank_fusion([dense, keyword])
        hybrid = merged[:CANDIDATES]  # cut to 20 before re-ranking
        return {
            "dense": dense,
            "keyword": keyword,
            "hybrid": hybrid,
            "dense+rerank": self.rerank(query, dense),
            "hybrid+rerank": self.rerank(query, hybrid),
            "hybrid-wide+rerank": self.rerank(query, merged),  # re-rank everything merged (up to 40)
        }

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        scores = self.reranker.score(query, [h.text for h in hits])
        rescored = (replace(h, score=s) for h, s in zip(hits, scores, strict=True))
        return sorted(rescored, key=lambda h: h.score, reverse=True)

    def close(self) -> None:
        self.pool.close()
