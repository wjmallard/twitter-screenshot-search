"""tweet_activity tool — histogram of tweet counts over time."""

from ..core.db import get_conn
from .config import SEARCH_SIMILARITY_FLOOR
from .embedding import embed_texts, vec_literal
from .server import mcp

_VALID_GRANULARITIES = {"day", "week", "month", "year"}


@mcp.tool()
async def tweet_activity(
    query: str | None = None,
    users: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    granularity: str = "year",
    score: str | None = None,
) -> str:
    """Show tweet counts over time, bucketed by day/week/month/year.
    Use to find when a topic or user was most active before drilling in.

    Args:
        query: Optional semantic search query to filter by topic.
        users: Optional list of handles to filter by (e.g. ["someone"]).
        after: Only include tweets after this date (YYYY-MM-DD).
        before: Only include tweets before this date (YYYY-MM-DD).
        granularity: Bucket size — "day", "week", "month", or "year" (default).
        score: When query is provided, include similarity scores per bucket:
               "mean", "max", or None (default: no scores).
    """
    if granularity not in _VALID_GRANULARITIES:
        return f"Error: granularity must be one of: {', '.join(sorted(_VALID_GRANULARITIES))}"

    conditions = ["ocr_text_clean IS NOT NULL"]
    params: dict = {}
    vec = None

    if query:
        query_emb = embed_texts([query])[0]
        vec = vec_literal(query_emb)
        conditions.append("embedding IS NOT NULL")
        conditions.append(
            "1 - (embedding <=> %(vec)s::vector) >= %(floor)s"
        )
        params["vec"] = vec
        params["floor"] = SEARCH_SIMILARITY_FLOOR

    if after:
        conditions.append("COALESCE(tweet_time, created_at) >= %(after)s::date")
        params["after"] = after
    if before:
        conditions.append("COALESCE(tweet_time, created_at) < %(before)s::date")
        params["before"] = before
    if users:
        normalized = [u.lstrip("@").lower() for u in users]
        conditions.append("mentioned_users && %(users)s::text[]")
        params["users"] = normalized

    where = " AND ".join(conditions)

    # Build SELECT columns
    bucket = f"date_trunc(%(granularity)s, COALESCE(tweet_time, created_at))"
    params["granularity"] = granularity

    select_cols = [f"{bucket} AS bucket", "COUNT(*) AS cnt"]
    if query and score in ("mean", "max"):
        score_fn = "AVG" if score == "mean" else "MAX"
        select_cols.append(
            f"{score_fn}(1 - (embedding <=> %(vec)s::vector)) AS sim"
        )

    select = ", ".join(select_cols)

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT {select}
            FROM screenshots
            WHERE {where}
            GROUP BY bucket
            ORDER BY bucket
            """,
            params,
        ).fetchall()

    if not rows:
        parts = []
        if query:
            parts.append(f"query={query!r}")
        if users:
            parts.append(f"users={users}")
        return f"No activity found ({', '.join(parts) or 'no filters'})."

    # Find max count for bar scaling
    max_count = max(r[1] for r in rows)
    total = sum(r[1] for r in rows)
    bar_width = 20

    # Format output
    has_scores = query and score in ("mean", "max")
    lines = []

    # Header
    header_parts = [f"{total} tweets"]
    if query:
        header_parts.append(f"matching {query!r}")
    if users:
        header_parts.append(f"by {', '.join('@' + u for u in users)}")
    header_parts.append(f"({granularity}ly)")
    lines.append(" ".join(header_parts))
    lines.append("")

    for row in rows:
        bucket_date = row[0]
        count = row[1]

        if granularity == "day":
            label = bucket_date.strftime("%Y-%m-%d")
        elif granularity == "week":
            label = f"{bucket_date.strftime('%Y-%m-%d')}w"
        elif granularity == "month":
            label = bucket_date.strftime("%Y-%m")
        else:
            label = bucket_date.strftime("%Y")

        bar_len = round(count / max_count * bar_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len

        if has_scores:
            sim = row[2]
            lines.append(f"{label}  {count:>5}  sim={sim:.2f}  {bar}")
        else:
            lines.append(f"{label}  {count:>5}  {bar}")

    return "\n".join(lines)
