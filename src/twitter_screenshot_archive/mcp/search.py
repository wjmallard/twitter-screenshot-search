"""search_tweets tool."""

from ..core.db import get_conn
from .config import (
    DEFAULT_SEARCH_LIMIT,
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
    offset: int = 0,
    min_score: float = SEARCH_SIMILARITY_FLOOR,
    after: str | None = None,
    before: str | None = None,
    users: list[str] | None = None,
    sort: str = "relevance",
) -> str:
    """Find tweets about a topic. Semantic search — matches meaning, not exact
    words. Use sort=chronological for a filtered timeline.

    Args:
        query: Search query text.
        limit: Max results to return (default 10).
        offset: Number of results to skip (default 0). Use to paginate.
        min_score: Minimum similarity threshold (default 0.4). Raise for
                   precision, lower for recall. Set to 0 to disable.
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        users: Optional list of handles to filter by (e.g. ["someone"]).
               Union semantics — includes any tweet mentioning any listed user.
        sort: "relevance" (default, by similarity) or "chronological"
              (by tweet time, newest first).
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    query_emb = embed_texts([query])[0]
    vec = vec_literal(query_emb)

    conditions = [
        "embedding IS NOT NULL",
        "1 - (embedding <=> %(vec)s::vector) >= %(floor)s",
    ]
    params: dict = {
        "vec": vec,
        "limit": limit,
        "offset": offset,
        "floor": min_score,
    }

    if after:
        conditions.append("COALESCE(tweet_time, created_at) >= %(after)s::date")
        params["after"] = after
    if before:
        conditions.append("COALESCE(tweet_time, created_at) < %(before)s::date")
        params["before"] = before
    if users:
        normalized = [u.lstrip("@").lower() for u in users]
        conditions.append("mentioned_users && %(users)s::text[]")
        params["users"] = normalized

    where = " AND ".join(conditions)

    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM screenshots WHERE {where}",
            params,
        ).fetchone()[0]

        if sort == "chronological":
            order_by = "COALESCE(tweet_time, created_at) DESC"
        else:
            order_by = "embedding <=> %(vec)s::vector"

        rows = conn.execute(
            f"""
            SELECT id, ocr_text_clean, tweet_time, mentioned_users,
                   1 - (embedding <=> %(vec)s::vector) AS similarity
            FROM screenshots
            WHERE {where}
            ORDER BY {order_by}
            LIMIT %(limit)s
            OFFSET %(offset)s
            """,
            params,
        ).fetchall()

    if not rows:
        return "No results found."

    sort_label = "newest first" if sort == "chronological" else "best match first"
    start = offset + 1
    end = offset + len(rows)
    if len(rows) < total - offset:
        header = f"Results {start}–{end} of {total} for: {query} ({sort_label})"
    else:
        header = f"Found {total} results for: {query} ({sort_label})"
    parts = [header + "\n"]
    for i, row in enumerate(rows, start):
        parts.append(_format_result(i, total, row))

    return "\n\n".join(parts)
