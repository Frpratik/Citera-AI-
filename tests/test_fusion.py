from app.repositories.chunks import SearchHit
from app.retrieval.fusion import reciprocal_rank_fusion


def hit(chunk_id: str) -> SearchHit:
    return SearchHit(chunk_id, "doc", "doc.pdf", 1, f"text {chunk_id}", 0.0)


def ids(hits: list[SearchHit]) -> list[str]:
    return [h.chunk_id for h in hits]


def test_chunk_high_in_both_lists_wins():
    meaning = [hit("a"), hit("b"), hit("c")]
    keyword = [hit("b"), hit("d"), hit("e")]
    # "b" is 2nd + 1st; "a" is 1st in only one list. Appearing in both wins.
    assert ids(reciprocal_rank_fusion([meaning, keyword]))[0] == "b"


def test_chunk_found_by_only_one_list_is_kept_without_duplicates():
    merged = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("c"), hit("a")]])
    assert sorted(ids(merged)) == ["a", "b", "c"]
    assert ids(merged)[0] == "a"  # found by both


def test_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
