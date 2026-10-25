from app.ingestion.chunking import chunk_pages
from app.ingestion.models import Page

SENTENCES = " ".join(f"Sentence number {i} talks about fees." for i in range(60))


def test_chunks_respect_size_and_keep_page_numbers():
    pages = [Page(1, SENTENCES), Page(2, "Short page.")]
    chunks = chunk_pages(pages, "doc1", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[-1].page == 2 and chunks[-1].text == "Short page."


def test_chunk_ids_and_indexes_are_unique_and_sequential():
    chunks = chunk_pages([Page(1, SENTENCES)], "doc1", chunk_size=300, chunk_overlap=50)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "doc1-c0"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_pages([Page(1, SENTENCES)], "doc1", chunk_size=300, chunk_overlap=80)
    first, second = chunks[0].text, chunks[1].text
    assert any(second.startswith(first[-n:]) for n in range(20, 81))
