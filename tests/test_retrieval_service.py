import pytest

from app.repositories.chunks import ChunkRepository
from app.retrieval.service import RetrievalService
from tests.fakes import FakeReranker
from tests.helpers import store_document

TARGET = "Clause 7.3: a refund request must be filed within 14 days of purchase."
FILLERS = [
    "The warranty covers manufacturing defects.",
    "Shipping takes five working days.",
    "Support is open on weekdays.",
    "Prices include tax.",
    "Gift cards never expire.",
]


class ScriptedEmbedder:
    """Meaning search that is WRONG on purpose: the question points at 'direction 0', the
    filler chunks point there too, but the chunk containing 7.3 points at 'direction 1'.
    So meaning search ranks the 7.3 chunk dead last."""

    def _unit(self, i: int) -> list[float]:
        v = [0.0] * 384
        v[i] = 1.0
        return v

    def embed_documents(self, texts):
        return [self._unit(1 if "7.3" in t else 0) for t in texts]

    def embed_query(self, text):
        return self._unit(0)


def make_service(db_pool, **overrides):
    settings = {"candidates": 20, "top_k": 5, "min_score": 0.1} | overrides
    return RetrievalService(
        ChunkRepository(db_pool), ScriptedEmbedder(), FakeReranker(), **settings
    )


def test_keyword_search_rescues_a_chunk_that_meaning_search_missed(db_pool):
    store_document(db_pool, ScriptedEmbedder(), "alice-owner", "terms.pdf", [*FILLERS, TARGET])
    question = "What does clause 7.3 say?"

    # Meaning search alone, limited to 3 results, only sees fillers.
    dense_only = ChunkRepository(db_pool).vector_search("alice-owner", ScriptedEmbedder().embed_query(question), 3)
    assert TARGET not in [h.text for h in dense_only]

    # The full pipeline still finds it, because keyword search knows "7.3".
    results = make_service(db_pool, candidates=3).retrieve("alice-owner", question)
    assert results[0].text == TARGET


def test_results_are_sorted_by_reranker_score_and_capped_at_top_k(db_pool):
    store_document(db_pool, ScriptedEmbedder(), "alice-owner", "terms.pdf", [*FILLERS, TARGET])
    results = make_service(db_pool, top_k=2, min_score=0.0).retrieve("alice-owner", "refund request clause 7.3")
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].text == TARGET


def test_weak_chunks_are_dropped_so_nothing_relevant_means_empty(db_pool):
    store_document(db_pool, ScriptedEmbedder(), "alice-owner", "terms.pdf", [*FILLERS, TARGET])
    assert make_service(db_pool).retrieve("alice-owner", "bake chocolate cake") == []


def test_a_chunk_the_reranker_dislikes_is_still_kept_if_meaning_search_clearly_likes_it(db_pool):
    # The fillers share NO words with the question (re-rank score 0) but are as close in
    # meaning as possible (similarity 1.0): the second signal keeps them.
    store_document(db_pool, ScriptedEmbedder(), "alice-owner", "terms.pdf", [*FILLERS, TARGET])
    question = "bake chocolate cake"
    assert make_service(db_pool, min_similarity=None).retrieve("alice-owner", question) == []
    kept = make_service(db_pool, min_similarity=0.3).retrieve("alice-owner", question)
    assert kept and all(hit.text in FILLERS for hit in kept)


def test_a_chunk_found_only_by_keyword_search_has_no_similarity_score_to_lean_on(db_pool):
    store_document(db_pool, ScriptedEmbedder(), "alice-owner", "terms.pdf", [*FILLERS, TARGET])
    # "refunds" finds the TARGET by keyword search (word forms match), but the fake re-ranker
    # sees no shared word (score 0) and meaning search never returned it (no similarity).
    # The fillers, close in meaning, are kept; the TARGET is not.
    service = make_service(db_pool, candidates=4, min_similarity=0.3)
    kept = [hit.text for hit in service.retrieve("alice-owner", "refunds")]
    assert kept and TARGET not in kept


def test_no_documents_means_empty(db_pool):
    assert make_service(db_pool).retrieve("alice-owner", "anything at all") == []


def test_other_owners_documents_are_never_returned(db_pool):
    store_document(db_pool, ScriptedEmbedder(), "alice-owner", "terms.pdf", [*FILLERS, TARGET])
    assert make_service(db_pool).retrieve("bob-owner", "clause 7.3 refund") == []


@pytest.mark.parametrize("min_score, expect_results", [(0.0, True), (0.9, False)])
def test_min_score_controls_how_strict_the_relevance_check_is(db_pool, min_score, expect_results):
    store_document(db_pool, ScriptedEmbedder(), "alice-owner", "terms.pdf", [*FILLERS, TARGET])
    # Fake reranker score for TARGET is 2 of 3 question words = 0.67.
    results = make_service(db_pool, min_score=min_score).retrieve("alice-owner", "refund clause cake")
    assert bool(results) is expect_results
