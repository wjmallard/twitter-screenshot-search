"""Explore tools — summarize_period, list_topics."""

import numpy as np

from .clustering import _cluster, _fetch_relevant
from .config import (
    SNIPPET_MAX_CHARS,
)
from .server import mcp


def _pick_snippets(medoid: dict, members: list[dict], max_snippets: int) -> list[dict]:
    """Pick up to max_snippets members: most similar to medoid, sorted chronologically."""
    members = [m for m in members if m["id"] != medoid["id"]]
    if len(members) <= max_snippets:
        selected = members
    else:
        medoid_vec = medoid["embedding"]
        medoid_norm = np.linalg.norm(medoid_vec)
        if medoid_norm > 0:
            medoid_vec = medoid_vec / medoid_norm

        scored = []
        for m in members:
            vec = m["embedding"]
            norm = np.linalg.norm(vec)
            sim = float(np.dot(medoid_vec, vec / norm)) if norm > 0 else 0.0
            scored.append((sim, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [m for _, m in scored[:max_snippets]]

    selected.sort(key=lambda r: r["tweet_time"] or r["created_at"])
    return selected


@mcp.tool()
async def summarize_period(
    after: str | None = None,
    before: str | None = None,
    topics: list[str] | None = None,
    max_topics: int = 10,
) -> str:
    """Cluster and summarize what happened in a time window or around
    specific topics. Returns rich detail per topic: date span, top users,
    representative snippets. Use for overviews, discourse tracing, and
    narrative reconstruction.

    Args:
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        topics: Optional topic strings to filter by (e.g. ["AI", "China trade"]).
        max_topics: Maximum number of topic clusters to return (default 10).
    """
    if not after and not before and not topics:
        return "Error: provide at least one of after/before date range or topics."

    rows = await _fetch_relevant(after=after, before=before, topics=topics)
    if not rows:
        return "No tweets found in the specified range."

    clusters = _cluster(rows, max_topics=max_topics)
    if not clusters:
        return "No tweet clusters formed — too few tweets."

    parts = []
    for i, c in enumerate(clusters, 1):
        start_date = c["start_date"].strftime("%Y-%m-%d")
        end_date = c["end_date"].strftime("%Y-%m-%d")
        header = f"--- Topic {i} ({c['count']} tweets, {start_date} — {end_date}) ---"
        lines = [header]

        if c["top_users"]:
            lines.append("Top users: " + ", ".join("@" + u for u in c["top_users"]))

        medoid_snippet = (c["medoid"]["ocr_text_clean"] or "")[:SNIPPET_MAX_CHARS]
        lines.append(f"Representative: {medoid_snippet}")

        snippet_rows = _pick_snippets(c["medoid"], c["members"], max_snippets=5)
        lines.append("")
        for row in snippet_rows:
            t = (row["tweet_time"] or row["created_at"]).isoformat()
            snippet = (row["ocr_text_clean"] or "")[:SNIPPET_MAX_CHARS]
            lines.append(f"[ID {row['id']}] {t} | {snippet}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


@mcp.tool()
async def list_topics(
    after: str | None = None,
    before: str | None = None,
    max_topics: int = 10,
) -> str:
    """List the main topics in a time window, ranked by tweet count.
    Lightweight overview — use summarize_period for detail.

    Args:
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        max_topics: Maximum number of topics to return (default 10).
    """
    if not after and not before:
        return "Error: provide at least one of after or before."

    rows = await _fetch_relevant(after=after, before=before)
    if not rows:
        return "No tweets found in the specified range."

    clusters = _cluster(rows, max_topics=max_topics)
    if not clusters:
        return "No tweet clusters formed — too few tweets."

    lines = []
    for i, c in enumerate(clusters, 1):
        start_date = c["start_date"].strftime("%b %d")
        end_date = c["end_date"].strftime("%b %d")
        medoid_snippet = (c["medoid"]["ocr_text_clean"] or "")[:200]
        lines.append(f"{i}. {medoid_snippet} ({c['count']} tweets, {start_date}–{end_date})")

    return "\n".join(lines)
