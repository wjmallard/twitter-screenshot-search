"""Shared clustering internals for summarize_period and list_topics."""

import json

import httpx
import numpy as np

from ..db import get_conn
from .config import CLUSTER_MIN_SIZE, TOPIC_SIM_THRESHOLD_PCT
from .embedding import embed_texts


async def _fetch_relevant(
    after: str | None = None,
    before: str | None = None,
    topics: list[str] | None = None,
) -> list[dict]:
    """Fetch rows with parsed embeddings, optional date/topic filtering.

    At least one of date range or topics must be provided (enforced by caller).
    """
    conditions = ["embedding IS NOT NULL"]
    params: dict = {}

    if after:
        conditions.append("COALESCE(tweet_time, created_at) >= %(after)s::date")
        params["after"] = after
    if before:
        conditions.append("COALESCE(tweet_time, created_at) < %(before)s::date")
        params["before"] = before

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

    if topics and rows:
        async with httpx.AsyncClient() as client:
            topic_embeddings = await embed_texts(client, topics)

        topic_vecs = np.array(topic_embeddings, dtype=np.float32)
        # Normalize topic vectors
        topic_norms = np.linalg.norm(topic_vecs, axis=1, keepdims=True)
        topic_vecs = topic_vecs / np.where(topic_norms == 0, 1, topic_norms)

        # Build row embedding matrix and normalize
        row_vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
        row_norms = np.linalg.norm(row_vecs, axis=1, keepdims=True)
        row_vecs = row_vecs / np.where(row_norms == 0, 1, row_norms)

        # Cosine similarity: each row vs each topic, take max across topics
        sim_matrix = row_vecs @ topic_vecs.T  # (N, num_topics)
        max_sims = sim_matrix.max(axis=1)  # (N,)

        # Keep top TOPIC_SIM_THRESHOLD_PCT of the distribution
        threshold = np.quantile(max_sims, 1.0 - TOPIC_SIM_THRESHOLD_PCT)
        filtered = [r for r, s in zip(rows, max_sims) if s >= threshold]

        # Don't discard below cluster_min_size — return everything rather
        # than losing rows that the caller can still use as a single group
        rows = filtered if len(filtered) >= CLUSTER_MIN_SIZE else rows

    return rows
