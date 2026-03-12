"""Orientation tools — cheap, fast, no embeddings."""

from datetime import datetime, timezone

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
