"""MCP server for semantic search over the Twitter screenshot archive.

Exposes a search_tweets tool that embeds queries via LM Studio's
/v1/embeddings endpoint and searches pgvector cosine similarity.

Fully separate from the Flask web app — no shared state, no imports
from app.py or minhash.py.
"""

import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

logging.getLogger("httpx").setLevel(logging.WARNING)
import yaml
from mcp.server.fastmcp import FastMCP
from tqdm import tqdm

from .db import get_conn

# ---------------------------------------------------------------------------
# Config — MCP-specific keys from config.yaml, with sensible defaults
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

try:
    with open(_CONFIG_PATH) as f:
        _raw = yaml.safe_load(f) or {}
except FileNotFoundError:
    _raw = {}

LMSTUDIO_URL = _raw.get("lmstudio_url", "http://localhost:1234")
EMBEDDING_MODEL = _raw.get("embedding_model", "text-embedding-embeddinggemma-300m")
EMBEDDING_DIM = 768
BACKFILL_BATCH_SIZE = _raw.get("embedding_batch_size", 64)
DEFAULT_SEARCH_LIMIT = _raw.get("embedding_search_limit", 10)
SNIPPET_MAX_CHARS = 300

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


async def _embed_texts(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Call LM Studio's OpenAI-compatible embeddings endpoint."""
    resp = await client.post(
        f"{LMSTUDIO_URL}/v1/embeddings",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [d["embedding"] for d in data]


def _vec_literal(embedding: list[float]) -> str:
    """Format a float list as a pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------


async def _check_lmstudio():
    """Verify LM Studio is reachable and the embedding model is loaded."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{LMSTUDIO_URL}/v1/models", timeout=5.0)
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        print(
            f"Error: Cannot reach LM Studio at {LMSTUDIO_URL}\n"
            "Make sure LM Studio is running with the embedding model loaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    models = [m["id"] for m in resp.json().get("data", [])]
    if EMBEDDING_MODEL not in models:
        print(
            f"Error: Model '{EMBEDDING_MODEL}' not loaded in LM Studio.\n"
            f"Available models: {', '.join(models) or '(none)'}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Backfill — runs once at startup
# ---------------------------------------------------------------------------


async def _backfill_embeddings():
    """Embed all rows that have ocr_text but no embedding yet."""
    with get_conn() as conn:
        pending = conn.execute(
            "SELECT count(*) FROM screenshots "
            "WHERE ocr_text_clean IS NOT NULL AND ocr_text_clean != '' AND embedding IS NULL"
        ).fetchone()[0]

        if pending == 0:
            return

        progress = tqdm(total=pending, desc="Embedding backfill", file=sys.stderr)

        async with httpx.AsyncClient() as client:
            while True:
                rows = conn.execute(
                    "SELECT id, ocr_text_clean FROM screenshots "
                    "WHERE ocr_text_clean IS NOT NULL AND ocr_text_clean != '' AND embedding IS NULL "
                    "ORDER BY id LIMIT %s",
                    (BACKFILL_BATCH_SIZE,),
                ).fetchall()

                if not rows:
                    break

                ids = [r[0] for r in rows]
                texts = [r[1] for r in rows]

                try:
                    embeddings = await _embed_texts(client, texts)
                except httpx.ConnectError:
                    progress.close()
                    print(
                        f"Cannot reach LM Studio at {LMSTUDIO_URL} — "
                        "backfill aborted, will retry next startup.",
                        file=sys.stderr,
                    )
                    return

                for row_id, emb in zip(ids, embeddings):
                    conn.execute(
                        "UPDATE screenshots SET embedding = %s::vector WHERE id = %s",
                        (_vec_literal(emb), row_id),
                    )
                conn.commit()
                progress.update(len(ids))

        progress.close()


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(server: FastMCP):
    await _check_lmstudio()
    await _backfill_embeddings()
    print("MCP server ready. Press Ctrl+D to exit.", file=sys.stderr)
    yield {}


mcp = FastMCP("twitter-archive", lifespan=_lifespan)


def _format_result(index: int, total: int, row) -> str:
    """Format a single search result as plain text."""
    id, text, tweet_time, mentioned, sim = row
    lines = [f"[{index}/{total}] ID {id} | sim: {sim:.2f}"]
    if tweet_time:
        lines[0] += f" | {tweet_time.isoformat()}"
    if mentioned:
        lines.append(f"Users: {', '.join('@' + u for u in mentioned)}")
    snippet = (text or "")[:SNIPPET_MAX_CHARS]
    if snippet:
        lines.append(snippet)
    return "\n".join(lines)


@mcp.tool()
async def search_tweets(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Search the Twitter screenshot archive by semantic similarity.

    Embeds the query and finds screenshots whose OCR text is closest
    in meaning.  Returns matching tweets with text, timestamps, and
    metadata.  Use a small limit first and increase if needed.
    Use get_tweet(id) to read the full text of any result.

    Args:
        query: Search query text.
        limit: Max results to return (default 10).
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
    """
    limit = max(1, min(limit, 200))

    try:
        async with httpx.AsyncClient() as client:
            query_emb = (await _embed_texts(client, [query]))[0]
    except httpx.ConnectError:
        return f"Error: Cannot reach LM Studio at {LMSTUDIO_URL}"

    vec = _vec_literal(query_emb)

    conditions = ["embedding IS NOT NULL"]
    params: dict = {
        "vec": vec,
        "limit": limit,
    }

    if after:
        conditions.append("COALESCE(tweet_time, created_at) >= %(after)s::date")
        params["after"] = after
    if before:
        conditions.append("COALESCE(tweet_time, created_at) < %(before)s::date")
        params["before"] = before

    where = " AND ".join(conditions)

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT id, ocr_text_clean, tweet_time, mentioned_users,
                   1 - (embedding <=> %(vec)s::vector) AS similarity
            FROM screenshots
            WHERE {where}
            ORDER BY embedding <=> %(vec)s::vector
            LIMIT %(limit)s
            """,
            params,
        ).fetchall()

    if not rows:
        return "No results found."

    parts = [f"Found {len(rows)} results for: {query}\n"]
    for i, row in enumerate(rows, 1):
        parts.append(_format_result(i, len(rows), row))

    return "\n\n".join(parts)


@mcp.tool()
async def get_tweet(id: int) -> str:
    """Get the full OCR text of a specific tweet screenshot by ID.

    Use this after search_tweets to read the complete text of
    a result that looks interesting.
    """
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, ocr_text_clean, tweet_time, mentioned_users
            FROM screenshots
            WHERE id = %(id)s
            """,
            {"id": id},
        ).fetchone()

    if not row:
        return f"No screenshot with id {id}"

    id, text, tweet_time, mentioned = row
    lines = [f"Tweet ID {id}"]
    if tweet_time:
        lines.append(f"Tweet time: {tweet_time.isoformat()}")
    if mentioned:
        lines.append(f"Users: {', '.join('@' + u for u in mentioned)}")
    lines.append("")
    lines.append(text or "(no text)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
