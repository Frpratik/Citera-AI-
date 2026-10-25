from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from tests.helpers import store_document


# ---------------------------------------------------------------- meaning search
def test_vector_search_ranks_most_similar_chunk_first(db_pool, embedder):
    store_document(
        db_pool, embedder, "alice-owner", "rules.pdf",
        ["the exam fee payment deadline is in march", "hostel rooms are allotted by merit"],
    )
    query = embedder.embed_query("exam fee deadline")
    hits = ChunkRepository(db_pool).vector_search("alice-owner", query, 5)
    assert hits[0].text.startswith("the exam fee")
    assert hits[0].score > hits[1].score
    assert hits[0].document_name == "rules.pdf" and hits[0].page == 1


def test_vector_search_never_returns_another_owners_chunks(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "a.pdf", ["secret salary information"])
    query = embedder.embed_query("salary")
    assert ChunkRepository(db_pool).vector_search("bob-owner", query, 5) == []


def test_vector_search_ignores_documents_that_are_not_ready(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "a.pdf", ["half processed text"], ready=False)
    query = embedder.embed_query("text")
    assert ChunkRepository(db_pool).vector_search("alice-owner", query, 5) == []


def test_deleting_a_document_removes_its_chunks(db_pool, embedder):
    record = store_document(db_pool, embedder, "alice-owner", "a.pdf", ["some text here"])
    assert DocumentRepository(db_pool).delete("alice-owner", record.id) is True
    query = embedder.embed_query("text")
    assert ChunkRepository(db_pool).vector_search("alice-owner", query, 5) == []


def test_delete_requires_ownership(db_pool, embedder):
    record = store_document(db_pool, embedder, "alice-owner", "a.pdf", ["some text here"])
    assert DocumentRepository(db_pool).delete("bob-owner", record.id) is False
    assert DocumentRepository(db_pool).get("alice-owner", record.id) is not None


def test_count_ready_counts_only_this_owners_ready_documents(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "a.pdf", ["one"])
    store_document(db_pool, embedder, "alice-owner", "b.pdf", ["two"], ready=False)
    store_document(db_pool, embedder, "bob-owner", "c.pdf", ["three"])
    docs = DocumentRepository(db_pool)
    assert docs.count_ready("alice-owner") == 1
    assert docs.count_ready("nobody-here") == 0


# ---------------------------------------------------------------- keyword search
CLAUSES = [
    "Clause 7.3: a refund request must be filed within 14 days of purchase.",
    "Clause 7.4: a refund may be denied if the product has been used.",
    "Clause 9.1: the warranty covers manufacturing defects for one year.",
]


def test_keyword_search_finds_an_exact_number(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "terms.pdf", CLAUSES)
    hits = ChunkRepository(db_pool).keyword_search("alice-owner", "7.3", 5)
    assert [h.text[:10] for h in hits] == ["Clause 7.3"]


def test_keyword_search_matches_word_forms(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "terms.pdf", ["Refunds are processed quickly."])
    assert ChunkRepository(db_pool).keyword_search("alice-owner", "refund", 5)  # stemming


def test_keyword_search_does_not_need_every_word_to_match(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "terms.pdf", CLAUSES)
    hits = ChunkRepository(db_pool).keyword_search("alice-owner", "What does clause 7.3 say?", 5)
    assert hits[0].text.startswith("Clause 7.3")  # "say" is in no chunk, still found


def test_keyword_search_with_only_stop_words_returns_nothing(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "terms.pdf", CLAUSES)
    assert ChunkRepository(db_pool).keyword_search("alice-owner", "what is the", 5) == []


def test_keyword_search_respects_owner_and_status(db_pool, embedder):
    store_document(db_pool, embedder, "alice-owner", "a.pdf", ["secret salary information"])
    store_document(db_pool, embedder, "alice-owner", "b.pdf", ["draft salary notes"], ready=False)
    repo = ChunkRepository(db_pool)
    assert len(repo.keyword_search("alice-owner", "salary", 5)) == 1  # ready only
    assert repo.keyword_search("bob-owner", "salary", 5) == []  # not Bob's
