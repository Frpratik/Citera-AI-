-- One row per chat question: how it went, how long each step took, how many tokens it used.
-- It never stores the question or the answer text.
CREATE TABLE chat_traces (
    id            bigserial PRIMARY KEY,
    owner_id      text NOT NULL,
    request_id    text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    route         text,
    outcome       text NOT NULL,            -- the answer type, or "error:<ErrorType>"
    total_ms      integer NOT NULL,
    llm_calls     integer NOT NULL,
    input_tokens  integer NOT NULL,
    output_tokens integer NOT NULL,
    fallback_used boolean NOT NULL DEFAULT false,
    spans         jsonb NOT NULL DEFAULT '[]',   -- [{"name": "...", "ms": 123}, ...]
    events        jsonb NOT NULL DEFAULT '[]'    -- e.g. ["fallback:openai/gpt-oss-20b"]
);
CREATE INDEX chat_traces_owner_idx ON chat_traces (owner_id, created_at);
