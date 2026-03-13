"""MCP server for the Twitter screenshot archive.

Subpackage layout:
    clustering.py  — shared clustering internals (PCA + HDBSCAN)
    config.py      — MCP-specific configuration
    drill.py       — get_tweet, browse_timeline, find_related, search_by_user tools
    embedding.py   — LM Studio embedding helpers, backfill
    explore.py     — summarize_period, list_topics, top_users, similar_users tools
    orient.py      — now, archive_range, count_screenshots tools
    search.py      — search_tweets tool
    server.py      — FastMCP instance, lifespan, entry point
"""

# Import tool modules to trigger @mcp.tool() registration
from . import drill
from . import explore
from . import orient
from . import search
from .server import main

__all__ = ["main"]
