"""Drill tools — get_tweet, browse_timeline, find_related, search_by_user."""

from ..core.db import get_conn
from ..core.minhash import query_related
from . import server
from .config import (
    SNIPPET_MAX_CHARS,
)

mcp = server.mcp


@mcp.tool()
async def get_tweet(id: int) -> str:
    """Get the full OCR text of a specific tweet by ID."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, ocr_text_clean, tweet_time, mentioned_users
            FROM screenshots
            WHERE id = %(id)s
            """,
            {
                "id": id,
            },
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


def _format_row(row) -> str:
    """Format a timeline row as plain text."""
    id, text, tweet_time, mentioned = row
    parts = [f"[ID {id}]"]
    if tweet_time:
        parts.append(tweet_time.isoformat())
    if mentioned:
        parts.append(", ".join("@" + u for u in mentioned))
    snippet = (text or "(no text)")[:SNIPPET_MAX_CHARS]
    parts.append(snippet)
    return " | ".join(parts)


@mcp.tool()
async def browse_timeline(
    id: int,
    before: int = 5,
    after: int = 5,
) -> str:
    """Browse tweets chronologically around a specific tweet. Not a search —
    just shows what was nearby in time.

    Args:
        id: Screenshot ID to center on.
        before: Number of earlier screenshots to show (default 5).
        after: Number of later screenshots to show (default 5).
    """
    with get_conn() as conn:
        focal = conn.execute(
            """
            SELECT id, ocr_text_clean, tweet_time, mentioned_users, created_at
            FROM screenshots
            WHERE id = %(id)s
            """,
            {
                "id": id,
            },
        ).fetchone()

        if not focal:
            return f"No screenshot with id {id}"

        focal_time = focal[4]  # created_at
        if focal_time is None:
            return _format_row(focal[:4])

        before_rows = conn.execute(
            """
            SELECT id, ocr_text_clean, tweet_time, mentioned_users
            FROM screenshots
            WHERE (created_at, id) < (%(ts)s, %(id)s)
              AND created_at IS NOT NULL
            ORDER BY created_at DESC, id DESC
            LIMIT %(limit)s
            """,
            {
                "ts": focal_time,
                "id": id,
                "limit": before,
            },
        ).fetchall()

        after_rows = conn.execute(
            """
            SELECT id, ocr_text_clean, tweet_time, mentioned_users
            FROM screenshots
            WHERE (created_at, id) > (%(ts)s, %(id)s)
              AND created_at IS NOT NULL
            ORDER BY created_at ASC, id ASC
            LIMIT %(limit)s
            """,
            {
                "ts": focal_time,
                "id": id,
                "limit": after,
            },
        ).fetchall()

    lines = []
    for row in reversed(before_rows):
        lines.append(_format_row(row))
    lines.append(f">>> {_format_row(focal[:4])} <<<")
    for row in after_rows:
        lines.append(_format_row(row))

    return "\n".join(lines)


@mcp.tool()
async def find_related(id: int, limit: int = 10) -> str:
    """Find tweets with similar wording to a specific tweet. Use to find other
    parts of the same thread, conversation, or reply chain.

    Args:
        id: Screenshot ID to find related tweets for.
        limit: Maximum number of results (default 10).
    """
    matches = query_related(server._lsh, server._minhashes, id, top_n=limit)

    if not matches:
        return f"No related tweets found for ID {id}"

    match_ids = [mid for mid, _ in matches]
    sim_by_id = dict(matches)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, ocr_text_clean, tweet_time, mentioned_users
            FROM screenshots
            WHERE id = ANY(%(ids)s)
            """,
            {
                "ids": match_ids,
            },
        ).fetchall()

    row_by_id = {row[0]: row for row in rows}

    lines = []
    for mid in match_ids:
        row = row_by_id.get(mid)
        if not row:
            continue
        sim = sim_by_id[mid]
        parts = [f"[ID {mid}] sim={sim:.2f}"]
        if row[2]:  # tweet_time
            parts.append(row[2].isoformat())
        if row[3]:  # mentioned_users
            parts.append(", ".join("@" + u for u in row[3]))
        snippet = (row[1] or "(no text)")[:SNIPPET_MAX_CHARS]
        parts.append(snippet)
        lines.append(" | ".join(parts))

    return "\n".join(lines)


@mcp.tool()
async def search_by_user(
    handle: str,
    limit: int = 20,
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Find tweets mentioning a specific user. Use to follow what @someone was
    saying or being discussed.

    Args:
        handle: Twitter handle to search for (with or without @).
        limit: Max results to return (default 20).
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
    """
    handle = handle.lstrip("@").lower()
    limit = max(1, min(limit, 200))

    conditions = ["%(handle)s = ANY(mentioned_users)"]
    params: dict = {
        "handle": handle,
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
            SELECT id, ocr_text_clean, tweet_time, mentioned_users
            FROM screenshots
            WHERE {where}
            ORDER BY COALESCE(tweet_time, created_at)
            LIMIT %(limit)s
            """,
            params,
        ).fetchall()

    if not rows:
        return f"No tweets found mentioning @{handle}"

    lines = [f"Found {len(rows)} tweets mentioning @{handle}\n"]
    for row in rows:
        lines.append(_format_row(row))

    return "\n".join(lines)
