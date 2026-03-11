CREATE DATABASE twitter_screenshot_archive;

\c twitter_screenshot_archive

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE screenshots (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    ocr_text TEXT,
    ocr_text_clean TEXT,
    created_at TIMESTAMPTZ,
    created_at_local TIMESTAMP,
    timezone TEXT,
    width INT,
    height INT,
    file_size BIGINT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ocr_text_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', COALESCE(ocr_text, ''))) STORED,
    minhash_signature BYTEA,
    mentioned_users TEXT[],
    tweet_time TIMESTAMPTZ,
    tweet_time_source TEXT
);

CREATE INDEX idx_screenshots_tsv ON screenshots USING GIN (ocr_text_tsv);
CREATE INDEX idx_screenshots_trgm ON screenshots USING GIN (ocr_text gin_trgm_ops);
CREATE INDEX idx_screenshots_created ON screenshots (created_at);
CREATE INDEX idx_screenshots_created_local ON screenshots (created_at_local);
CREATE INDEX idx_screenshots_mentioned_users ON screenshots USING GIN (mentioned_users);
CREATE INDEX idx_screenshots_tweet_time ON screenshots (tweet_time);
