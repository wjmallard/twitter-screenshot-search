# Twitter Archive

A Twitter screenshot archive with full-text search and similarity browsing. OCR-indexes images into Postgres and serves them via a local search UI with fuzzy matching and MinHash-based related-tweet discovery.

*Vibe-coded with Claude Code.*

## Overview

1. **Ingest** — OCR screenshots with Tesseract, compute MinHash signatures, insert into Postgres
2. **Search** — Flask web UI with fuzzy search, related-tweet discovery, and image lightbox

Dates are parsed from JSON sidecars with EXIF fallback.

## Stack

- Python 3.12
- PostgreSQL — with `tsvector` for full-text search, and `pg_trgm` for trigram search
- Tesseract — for OCR
- OpenCV — for CLAHE preprocessing
- datasketch — with MinHash/LSH for related-tweet similarity search
- Flask — with Jinja2 templates

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL
- Tesseract

### Configure Database

```bash
psql -f sql/schema.sql
```

Expects local trust authentication (no password).

### Configure Settings

```bash
cp config.yaml.example config.yaml
```

### Install Dependencies

```bash
uv sync
```

## Usage

### Ingest

```bash
uv run ingest
```

1. Walk the screenshot directory
2. Pre-process with CLAHE (to enhance contrast of dark-mode screenshots)
3. Run Tesseract for OCR
4. Insert into Postgres

- Skips previously-ingested files to support incremental updates and graceful restart after Ctrl+C.
- Displays a `tqdm` progress bar.

### Search

```bash
uv run search
```

Runs a local webserver at `http://localhost:5000`.

**Fuzzy search modes:**
- **Word** (default) — full-text search with stemming. "running" matches "run". Supports boolean syntax.
- **Char** — character-level trigram matching. Catches typos and OCR errors.
- **None** — exact substring match (case-insensitive).

**Sort options:**
- **Best** (default) — relevance × recency
- **Strongest** — relevance only
- **Newest** / **Oldest** — chronological

**Best** sorts by relevance weighted by recency. For example, with a 30-day half-life: a 30-day-old result scores half as much as an identical match from today, 60-day-old 1/3, 90-day-old 1/4, etc. Formula: `relevance_score × 1 / (age / half_life + 1)`. Half-life is set in `config.yaml`.

**Related tweets:** Click "related" on any result to find similar screenshots using MinHash/LSH (via [datasketch](https://github.com/ekzhu/datasketch)). OCR text is normalized, split into word 3-shingles, and hashed into 128-permutation MinHash signatures. An in-memory LSH index is built on server start for fast approximate nearest-neighbor lookup. Results are ranked by estimated Jaccard similarity.
