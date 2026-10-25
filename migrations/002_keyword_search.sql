-- Keyword search: Postgres fills this column itself from `content` (lowercased, stemmed,
-- stop words removed), so Python code never has to maintain it.
ALTER TABLE chunks
    ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- GIN = inverted index: word -> chunks that contain it (like the index at the back of a book).
CREATE INDEX chunks_content_tsv_gin ON chunks USING gin (content_tsv);
