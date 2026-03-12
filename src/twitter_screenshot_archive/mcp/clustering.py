"""Shared clustering internals for summarize_period and list_topics."""

import json

import httpx
import numpy as np

from ..db import get_conn
from .config import CLUSTER_MIN_SIZE, COARSE_SIMILARITY_FLOOR, TOPIC_SIM_THRESHOLD_PCT
from .embedding import embed_texts, vec_literal


def _parse_rows(db_rows: list) -> list[dict]:
    """Parse DB rows into dicts with numpy embeddings."""
    rows = []
    for r in db_rows:
        rows.append({
            "id": r[0],
            "ocr_text_clean": r[1],
            "tweet_time": r[2],
            "mentioned_users": r[3],
            "embedding": np.array(json.loads(r[4]), dtype=np.float32),
            "created_at": r[5],
        })
    return rows


async def _fetch_relevant(
    after: str | None = None,
    before: str | None = None,
    topics: list[str] | None = None,
) -> list[dict]:
    """Fetch rows with parsed embeddings, optional date/topic filtering.

    At least one of date range or topics must be provided (enforced by caller).

    When topics are provided, a two-pass filter runs:
    1. SQL coarse pre-filter via HNSW index (cosine sim >= SEARCH_SIMILARITY_FLOOR)
       with UNION ALL across topic vectors
    2. Numpy refinement: keep top TOPIC_SIM_THRESHOLD_PCT of the similarity
       distribution
    """
    date_conditions = []
    date_params: dict = {}

    if after:
        date_conditions.append("COALESCE(tweet_time, created_at) >= %(after)s::date")
        date_params["after"] = after
    if before:
        date_conditions.append("COALESCE(tweet_time, created_at) < %(before)s::date")
        date_params["before"] = before

    if topics:
        async with httpx.AsyncClient() as client:
            topic_embeddings = await embed_texts(client, topics)

        # Build UNION ALL across topic vectors — each branch gets its own
        # HNSW index scan for coarse pre-filtering
        date_where = (" AND " + " AND ".join(date_conditions)) if date_conditions else ""
        params: dict = dict(date_params)
        params["floor"] = COARSE_SIMILARITY_FLOOR

        branches = []
        for i, emb in enumerate(topic_embeddings):
            key = f"topic_{i}"
            params[key] = vec_literal(emb)
            branches.append(
                f"SELECT id, ocr_text_clean, tweet_time, mentioned_users, "
                f"embedding::text, created_at "
                f"FROM screenshots "
                f"WHERE embedding IS NOT NULL "
                f"AND 1 - (embedding <=> %({key})s::vector) >= %(floor)s"
                f"{date_where}"
            )

        sql = " UNION ALL ".join(branches)

        with get_conn() as conn:
            db_rows = conn.execute(sql, params).fetchall()

        # Dedup by id (UNION ALL across topics can return the same row twice)
        seen: set[int] = set()
        deduped = []
        for r in db_rows:
            if r[0] not in seen:
                seen.add(r[0])
                deduped.append(r)

        rows = _parse_rows(deduped)

        # Numpy refinement: top TOPIC_SIM_THRESHOLD_PCT
        if rows:
            topic_vecs = np.array(topic_embeddings, dtype=np.float32)
            topic_norms = np.linalg.norm(topic_vecs, axis=1, keepdims=True)
            topic_vecs = topic_vecs / np.where(topic_norms == 0, 1, topic_norms)

            row_vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
            row_norms = np.linalg.norm(row_vecs, axis=1, keepdims=True)
            row_vecs = row_vecs / np.where(row_norms == 0, 1, row_norms)

            sim_matrix = row_vecs @ topic_vecs.T  # (N, num_topics)
            max_sims = sim_matrix.max(axis=1)  # (N,)

            threshold = np.quantile(max_sims, 1.0 - TOPIC_SIM_THRESHOLD_PCT)
            filtered = [r for r, s in zip(rows, max_sims) if s >= threshold]

            # Don't discard below cluster_min_size — return everything rather
            # than losing rows that the caller can still use as a single group
            rows = filtered if len(filtered) >= CLUSTER_MIN_SIZE else rows

        return rows

    # Date-only path — no topic filtering, just fetch all embedded rows in range
    conditions = ["embedding IS NOT NULL"]
    conditions.extend(date_conditions)
    params = dict(date_params)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        db_rows = conn.execute(
            f"""
            SELECT id, ocr_text_clean, tweet_time, mentioned_users,
                   embedding::text, created_at
            FROM screenshots
            WHERE {where}
            """,
            params,
        ).fetchall()

    return _parse_rows(db_rows)
