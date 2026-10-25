-- A job queue kept in Postgres: no extra service to run, and a job is created in the
-- same database as the document it belongs to.
CREATE TABLE ingest_jobs (
    id           bigserial PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    status       text NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'done', 'failed')),
    attempts     integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL DEFAULT 3,
    run_after    timestamptz NOT NULL DEFAULT now(),  -- not before this time (retry delay)
    locked_at    timestamptz,                         -- when a worker picked it up
    last_error   text,                                -- the error TYPE only, never document text
    created_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz
);

-- Workers look for the next queued job, so index exactly those rows.
CREATE INDEX ingest_jobs_claim_idx ON ingest_jobs (run_after) WHERE status = 'queued';
CREATE INDEX ingest_jobs_document_idx ON ingest_jobs (document_id);

-- The database itself guarantees at most one waiting-or-running job per document.
CREATE UNIQUE INDEX ingest_jobs_one_active_per_document
    ON ingest_jobs (document_id) WHERE status IN ('queued', 'running');
