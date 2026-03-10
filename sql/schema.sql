CREATE DATABASE twitter_screenshot_search;

\c twitter_screenshot_search

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE screenshots (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    ocr_text TEXT,
    created_at TIMESTAMPTZ,
    created_at_local TIMESTAMP,
    timezone TEXT,
    width INT,
    height INT,
    file_size BIGINT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ocr_text_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', COALESCE(ocr_text, ''))) STORED
);

CREATE INDEX idx_screenshots_tsv ON screenshots USING GIN (ocr_text_tsv);
CREATE INDEX idx_screenshots_trgm ON screenshots USING GIN (ocr_text gin_trgm_ops);
CREATE INDEX idx_screenshots_created ON screenshots (created_at);
CREATE INDEX idx_screenshots_created_local ON screenshots (created_at_local);
