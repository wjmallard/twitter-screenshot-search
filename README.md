# Twitter Screenshot Search

Full-text search for a Twitter screenshot archive. OCR-indexes images into Postgres and serves them via a local fuzzy search UI.

*Vibe-coded with Claude Code.*

## Overview

1. **Ingest** — Run Tesseract and insert into Postgres database
2. **Search** — Flask web UI with fuzzy search

Dates are parsed from JSON sidecars with EXIF fallback.

## Stack

- Python 3.12
- PostgreSQL with `pg_trgm` for trigram search and `tsvector` for full-text search
- Tesseract OCR with OpenCV CLAHE preprocessing
- Flask with Jinja2 templates

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
