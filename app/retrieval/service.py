import logging
from dataclasses import replace

from app.embeddings.service import Embedder
from app.observability.tracing import span
from app.repositories.chunks import ChunkRepository, SearchHit
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class RetrievalService:
    """Question in, the few best chunks out (or an empty list = nothing relevant found).

        meaning search ──┐
                         ├─ merge (RRF) ─ top candidates ─ rerank ─ keep top_k ─ relevance check
        keyword search ──┘
    """

    def __init__(
        self,
        chunks: ChunkRepository,
        embedder: Embedder,
        reranker: Reranker,
        candidates: int = 20,
        top_k: int = 5,
        min_score: float = 0.1,
        min_similarity: float | None = None,
    ):
        self._chunks = chunks
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates
        self._top_k = top_k
        self._min_score = min_score
        self._min_similarity = min_similarity

    def retrieve(self, owner_id: str, question: str) -> list[SearchHit]:
        with span("retrieval.embed_question"):
            query_vector = self._embedder.embed_query(question)
        with span("retrieval.meaning_search"):
            by_meaning = self._chunks.vector_search(owner_id, query_vector, self._candidates)
        with span("retrieval.keyword_search"):
            by_keyword = self._chunks.keyword_search(owner_id, question, self._candidates)

        # The re-ranker sees EVERYTHING either search found (up to 2 x candidates). Cutting the
        # merged list to `candidates` first let noisy keyword hits push good meaning hits out
        # of the pool; the Phase 8 evaluation showed this on a larger corpus.
        merged = reciprocal_rank_fusion([by_meaning, by_keyword])
        if not merged:
            return []
        similarity = {hit.chunk_id: hit.score for hit in by_meaning}  # meaning-search scores

        # Replace the rough search score with the reranker's score, then sort by it.
        with span("retrieval.rerank"):
            scores = self._reranker.score(question, [hit.text for hit in merged])
        reranked = sorted(
            (replace(hit, score=score) for hit, score in zip(merged, scores, strict=True)),
            key=lambda hit: hit.score,
            reverse=True,
        )

        # Relevance check: junk is dropped, so the LLM is never asked to build an answer out of
        # text that has nothing to do with the question. Two signals, because the re-ranker
        # puts the right chunk FIRST but often gives it a low absolute score for everyday
        # wording: keep a chunk if the re-ranker likes it OR meaning search clearly does.
        # This is deliberately lenient: deciding "is it really answered?" is the job of the
        # answering step, which reads the text.
        def is_relevant(hit: SearchHit) -> bool:
            if hit.score >= self._min_score:
                return True
            return self._min_similarity is not None and similarity.get(hit.chunk_id, -1.0) >= self._min_similarity

        kept = [hit for hit in reranked[: self._top_k] if is_relevant(hit)]
        logger.info(
            "Retrieved: %d by meaning, %d by keyword, %d merged, %d kept",
            len(by_meaning),
            len(by_keyword),
            len(merged),
            len(kept),
        )
        return kept
