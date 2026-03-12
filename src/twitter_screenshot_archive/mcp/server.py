"""FastMCP instance, lifespan, and entry point."""

import pickle
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..db import get_conn, load_all_signatures, signature_fingerprint
from ..minhash import build_lsh_index
from .embedding import backfill_embeddings, check_lmstudio

_CACHE_DIR = Path.home() / ".cache" / "twitter-screenshot-archive"
_CACHE_FILE = _CACHE_DIR / "lsh_index.pkl"

_lsh = None
_minhashes = {}


def _init_lsh():
    """Build or load LSH index, using a pickle cache when possible."""
    global _lsh, _minhashes

    with get_conn() as conn:
        fingerprint = signature_fingerprint(conn)

    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "rb") as f:
                cached = pickle.load(f)
            if cached["fingerprint"] == fingerprint:
                _lsh, _minhashes = cached["lsh"], cached["minhashes"]
                print(
                    f"LSH index loaded from cache ({fingerprint[0]} signatures).",
                    file=sys.stderr,
                )
                return
            else:
                print("LSH cache stale, rebuilding...", file=sys.stderr)
        except Exception:
            print("LSH cache unreadable, rebuilding...", file=sys.stderr)

    with get_conn() as conn:
        sigs = load_all_signatures(conn)
    t0 = time.monotonic()
    print(f"Building LSH index from {len(sigs)} signatures...", file=sys.stderr)
    _lsh, _minhashes = build_lsh_index(sigs)
    elapsed = time.monotonic() - t0
    print(f"LSH index ready ({elapsed:.1f}s).", file=sys.stderr)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(
            {"fingerprint": fingerprint, "lsh": _lsh, "minhashes": _minhashes}, f
        )
    print(f"LSH cache saved to {_CACHE_FILE}", file=sys.stderr)


@asynccontextmanager
async def _lifespan(server: FastMCP):
    _init_lsh()
    await check_lmstudio()
    await backfill_embeddings()
    print("MCP server ready. Press Ctrl+D to exit.", file=sys.stderr)
    yield {}


mcp = FastMCP("twitter-archive", lifespan=_lifespan)


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    mcp.run(transport="stdio")
