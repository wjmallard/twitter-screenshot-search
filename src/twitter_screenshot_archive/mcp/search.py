"""search_tweets tool."""

import httpx

from ..db import get_conn
from .config import (
    DEFAULT_SEARCH_LIMIT,
    LMSTUDIO_URL,
    SEARCH_SIMILARITY_FLOOR,
    SNIPPET_MAX_CHARS,
)
from .embedding import embed_texts, vec_literal
from .server import mcp


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
    sort: str = "relevance",
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
        sort: "relevance" (default, by similarity) or "chronological"
              (by tweet time, oldest first).
    """
    limit = max(1, min(limit, 200))

    try:
        async with httpx.AsyncClient() as client:
            query_emb = (await embed_texts(client, [query]))[0]
    except httpx.ConnectError:
        return f"Error: Cannot reach LM Studio at {LMSTUDIO_URL}"

    vec = vec_literal(query_emb)

    conditions = [
        "embedding IS NOT NULL",
        "1 - (embedding <=> %(vec)s::vector) >= %(floor)s",
    ]
    params: dict = {
        "vec": vec,
        "limit": limit,
        "floor": SEARCH_SIMILARITY_FLOOR,
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
            ORDER BY {"COALESCE(tweet_time, created_at)" if sort == "chronological" else "embedding <=> %(vec)s::vector"}
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
