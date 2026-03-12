"""Orientation tools — cheap, fast, no embeddings."""

from datetime import datetime, timezone

from ..db import get_conn
from .server import mcp


@mcp.tool()
async def now() -> str:
    """Get the current date and time. Use this to resolve relative
    references like 'last week' or 'yesterday'."""
    utc = datetime.now(timezone.utc)
    local = datetime.now().astimezone()
    tz_name = local.tzinfo.tzname(local)
    return (
        f"UTC: {utc.isoformat()}\n"
        f"Local: {local.isoformat()} ({tz_name})"
    )


@mcp.tool()
async def archive_range() -> str:
    """Get the first and last dates in the archive."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                MIN(COALESCE(tweet_time, created_at)),
                MAX(COALESCE(tweet_time, created_at))
            FROM screenshots
            """
        ).fetchone()

    if not row or not row[0]:
        return "Archive is empty."

    return (
        f"First: {row[0].isoformat()}\n"
        f"Last: {row[1].isoformat()}"
    )
