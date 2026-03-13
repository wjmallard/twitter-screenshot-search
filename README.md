# Twitter Screenshot Archive

Full-text search and topic discovery over a personal archive of Twitter screenshots. OCR-indexes images into Postgres with two optional interfaces: a Flask web UI for fuzzy search and similarity browsing, and an MCP server for LLM-driven semantic search, topic clustering, and discourse tracing.

## Overview

1. **Ingest** — OCR screenshots with Tesseract, compute MinHash signatures, insert into Postgres
2. **Search** (optional) — Flask web UI with fuzzy search, related-tweet discovery, and image lightbox
3. **MCP Server** (optional) — Semantic search, topic clustering, and discourse tracing via LM Studio embeddings, exposed as MCP tools for LLM chat models (e.g. Qwen, Claude)

Dates are parsed from JSON sidecars with EXIF fallback.

## Stack

- Python 3.12
- PostgreSQL (pg_trgm, pgvector)
- Tesseract
- OpenCV
- datasketch
- Flask (optional — web UI)
- scikit-learn, MCP (optional — MCP server)

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL
  - pg_trgm
  - pgvector (optional — for [MCP server](#mcp-server-optional))
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
uv sync                          # ingest only (core)
uv sync --extra web              # + Flask search UI
uv sync --extra mcp              # + MCP server
uv sync --extra web --extra mcp  # everything
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

### Search (optional)

```bash
uv run web
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

### MCP Server (optional)

Exposes the archive to LLM chat models (e.g. Qwen, Claude) as MCP tools for semantic search, topic clustering, and discourse tracing.

#### Prerequisites

- [LM Studio](https://lmstudio.ai) running locally with an embedding model (default: `text-embedding-embeddinggemma-300m`)
- pgvector extension installed in PostgreSQL

#### Setup

Install with `uv sync --extra mcp` (see [Install Dependencies](#install-dependencies)).

#### Run

```bash
uv run mcp
```

On first run, embeds all OCR text via LM Studio. Subsequent starts catch up on new entries only.

#### Tools

Ten tools organized into three tiers:

**Orient** — cheap, fast, no embeddings:
- **`now()`** — Current date and time (UTC and local). Resolves relative references like "last week."
- **`archive_range()`** — First and last dates in the archive.
- **`count_screenshots(after?, before?)`** — Count screenshots in a time window.

**Explore** — embedding-based, discover structure:
- **`list_topics(after?, before?, max_topics?)`** — Lightweight table of contents: topic label + tweet count, ranked by size. Uses PCA dimensionality reduction and HDBSCAN clustering.
- **`summarize_period(after?, before?, topics?, max_topics?)`** — Rich clustered detail per topic: date span, tweet count, top mentioned users, and representative snippet. Supports topic filtering — pass topic strings to focus on specific themes. At least one of date range or topics required.
- **`search_tweets(query, limit?, after?, before?, sort?)`** — Semantic similarity search. Returns snippets ranked by relevance or chronologically. Supports date filtering.

**Drill** — follow threads once you have a foothold:
- **`find_related(id, limit?)`** — Lexically similar tweets via MinHash. Finds other parts of the same thread, conversation, or reply chain.
- **`browse_timeline(id, before?, after?)`** — Chronologically adjacent screenshots. Not a search — shows what was nearby in time.
- **`search_by_user(handle, limit?, after?, before?)`** — Tweets mentioning a specific @user, sorted chronologically.
- **`get_tweet(id)`** — Full OCR text of a specific screenshot.

#### Customization

Copy the example prompt file to provide personal context to the LLM:

```bash
cp mcp_prompt.txt.example mcp_prompt.txt
```

Edit `mcp_prompt.txt` with your interests, terminology, and preferences. This is injected into the MCP server instructions on startup — no need to edit system prompts per session. The file is gitignored.

#### MCP Client Configuration

```json
{
  "mcpServers": {
    "twitter-archive": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/twitter-screenshot-archive", "mcp"]
    }
  }
}
```

## Project Structure

```
src/twitter_screenshot_archive/
    core/           # Shared infrastructure: db, config, cleaning, ingest, minhash
    web/            # Flask search UI, templates, static assets
    mcp/            # MCP server, tools, clustering pipeline
```

Both `web/` and `mcp/` import from `core/`. Neither imports from the other. Each is independently installable via optional dependencies.
