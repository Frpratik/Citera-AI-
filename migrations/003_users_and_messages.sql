CREATE TABLE users (
    id            uuid PRIMARY KEY,
    email         text NOT NULL,
    password_hash text NOT NULL,   -- never the password itself
    created_at    timestamptz NOT NULL DEFAULT now()
);
-- One account per email, ignoring upper/lower case.
CREATE UNIQUE INDEX users_email_unique ON users (lower(email));

-- Chat history. owner_id holds the user's id (as text, like documents.owner_id).
CREATE TABLE messages (
    id          bigserial PRIMARY KEY,
    owner_id    text NOT NULL,
    role        text NOT NULL CHECK (role IN ('user', 'assistant')),
    content     text NOT NULL,
    answer_type text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX messages_owner_idx ON messages (owner_id, id);
