"""FastMCP instance, lifespan, and entry point."""

import signal
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from .embedding import backfill_embeddings, check_lmstudio


@asynccontextmanager
async def _lifespan(server: FastMCP):
    await check_lmstudio()
    await backfill_embeddings()
    print("MCP server ready. Press Ctrl+D to exit.", file=sys.stderr)
    yield {}


mcp = FastMCP("twitter-archive", lifespan=_lifespan)


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    mcp.run(transport="stdio")
