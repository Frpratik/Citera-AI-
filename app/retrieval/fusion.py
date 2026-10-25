from app.repositories.chunks import SearchHit


def reciprocal_rank_fusion(rankings: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """Merge several ranked lists into one, using only each chunk's POSITION.

    Scores from different searches (cosine vs keyword rank) are on different scales and
    cannot be added. Positions can: a chunk at rank r in a list earns 1 / (k + r) points,
    and points add up across lists. A chunk that is high in BOTH lists wins; a chunk
    found by only one list still stays in the merged list. k=60 is the usual default; it
    just stops the very first rank from dominating.
    """
    points: dict[str, float] = {}
    first_seen: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            points[hit.chunk_id] = points.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    best_first = sorted(points, key=lambda chunk_id: points[chunk_id], reverse=True)
    return [first_seen[chunk_id] for chunk_id in best_first]
