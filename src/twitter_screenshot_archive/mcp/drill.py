"""get_tweet tool."""

from ..db import get_conn
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
            WHERE id = %s
            """,
            (id,),
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
