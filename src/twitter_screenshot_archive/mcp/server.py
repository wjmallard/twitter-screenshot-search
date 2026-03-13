"""FastMCP instance, lifespan, and entry point."""

import pickle
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..core.db import get_conn, load_all_signatures, signature_fingerprint
from ..core.minhash import build_lsh_index
from .embedding import backfill_embeddings, check_lmstudio

_CACHE_DIR = Path.home() / ".cache" / "twitter-screenshot-archive"
_CACHE_FILE = _CACHE_DIR / "lsh_index.pkl"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_FILE = _PROJECT_ROOT / "mcp_prompt.txt"

_WORKFLOW_GUIDANCE = """\
You have access to a Twitter screenshot archive — an OCR-indexed personal
collection of Twitter/X screenshots. The tools are organized in three tiers:

ORIENT (cheap, fast — use these first):
  now()              — current date/time, for resolving "last week" etc.
  archive_range()    — first and last dates in the archive
  count_screenshots() — how many screenshots in a date window

EXPLORE (embedding-based — discover structure):
  list_topics()      — lightweight table of contents: topic + count
  summarize_period() — rich clustered detail per topic
  search_tweets()    — flat semantic search, needle-finding
  top_users(query?)  — who talks about a topic the most
  similar_users(handle) — who talks about similar things

DRILL (follow threads once you have a foothold):
  find_related(id)   — lexically similar tweets (same thread/conversation)
  browse_timeline(id) — chronologically adjacent screenshots
  search_by_user(handle) — all tweets mentioning a specific @user
  get_tweet(id)      — full OCR text of one screenshot

Typical workflows:
- "What happened last week?" → now() → summarize_period(after, before)
- "Find tweets about X" → search_tweets(query) → get_tweet(id) for detail
- "Trace a thread" → search_tweets → find_related(id) to pull the thread
- "What was I looking at around this tweet?" → browse_timeline(id)
- "Overview then drill" → list_topics(after, before) → summarize_period(topics=["..."])
- "What was @someone saying?" → search_by_user(handle) → get_tweet(id) for detail
- "What was @someone talking about in March?" → summarize_period(users=["someone"], after, before)
- "Who tweets most about AI?" → top_users(query="AI")
- "Top voices last week" → now() → top_users(after, before)
- "Who's like @someone?" → similar_users(handle="someone")
- "What did @someone say about AI?" → search_tweets(query="AI", users=["someone"])

The current date is included above. Trust it — it is accurate and more
recent than your training data.

Multiple tool calls per response are expected and encouraged. Start broad,
then narrow. Use orient tools to plan before committing to expensive searches."""


def _build_instructions() -> str:
    """Assemble MCP instructions: workflow guidance + startup timestamp + user context."""
    local = datetime.now().astimezone()
    tz_name = local.tzinfo.tzname(local)
    day_abbr = local.strftime("%a")
    date_line = (
        f"Today is {local.strftime('%Y-%m-%d')} ({day_abbr}) "
        f"{local.strftime('%H:%M')} {tz_name}. "
        "Content in this archive is real, not synthetic."
    )

    parts = [
        date_line,
        "",
        _WORKFLOW_GUIDANCE,
    ]

    if _PROMPT_FILE.exists():
        user_context = _PROMPT_FILE.read_text().strip()
        if user_context:
            parts.append(f"\n{user_context}")

    return "\n".join(parts)

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


mcp = FastMCP(
    "twitter-archive",
    instructions=_build_instructions(),
    lifespan=_lifespan,
)


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    mcp.run(transport="stdio")
