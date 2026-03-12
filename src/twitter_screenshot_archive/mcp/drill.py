"""Drill tools — get_tweet, browse_timeline."""

from ..db import get_conn
from .config import SNIPPET_MAX_CHARS
from .server import mcp


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
    """Browse screenshots chronologically around a specific tweet. Shows what
    was captured nearby in time — not a search, just browsing context.

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
