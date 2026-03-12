"""MCP server for the Twitter screenshot archive.

Subpackage layout:
    server.py      — FastMCP instance, lifespan, entry point
    config.py      — MCP-specific configuration
    embedding.py   — LM Studio embedding helpers, backfill
    search.py      — search_tweets tool
    drill.py       — get_tweet tool
"""

# Import tool modules to trigger @mcp.tool() registration
from . import drill
from . import search
from .server import main

__all__ = ["main"]
