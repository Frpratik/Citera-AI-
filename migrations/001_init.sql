CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          uuid PRIMARY KEY,
    owner_id    text NOT NULL,
    filename    text NOT NULL,
    status      text NOT NULL CHECK (status IN ('processing', 'ready', 'failed')),
    page_count  integer,
    chunk_count integer,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX documents_owner_idx ON documents (owner_id);

CREATE TABLE chunks (
    id          text PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    page        integer NOT NULL,
    chunk_index integer NOT NULL,
    content     text NOT NULL,
    embedding   vector(384) NOT NULL
);
CREATE INDEX chunks_document_idx ON chunks (document_id);

-- Approximate nearest-neighbour index for cosine distance.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
