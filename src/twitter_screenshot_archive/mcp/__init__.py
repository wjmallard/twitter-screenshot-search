"""MCP server for the Twitter screenshot archive.

Subpackage layout:
    server.py      — FastMCP instance, lifespan, entry point
    config.py      — MCP-specific configuration
    embedding.py   — LM Studio embedding helpers, backfill
    clustering.py  — shared clustering internals (PCA + HDBSCAN)
    orient.py      — now tool
    search.py      — search_tweets tool
    drill.py       — get_tweet, browse_timeline, find_related tools
    explore.py     — summarize_period, list_topics tools
"""

# Import tool modules to trigger @mcp.tool() registration
from . import drill
from . import explore
from . import orient
from . import search
from .server import main

__all__ = ["main"]
