CREATE TABLE IF NOT EXISTS logs (
    id         SERIAL PRIMARY KEY,
    prompt     TEXT        NOT NULL,
    result     SMALLINT    NOT NULL CHECK (result IN (0, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
