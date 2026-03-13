"""Shared clustering internals for summarize_period and list_topics."""

import json

import httpx
import numpy as np

from ..db import get_conn
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from .config import (
    CLUSTER_MIN_SAMPLES,
    CLUSTER_MIN_SIZE,
    COARSE_SIMILARITY_FLOOR,
    PCA_N_COMPONENTS,
    TIME_WEIGHT,
    TOPIC_SIM_THRESHOLD_PCT,
)
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

        sql = " UNION ".join(branches)

        with get_conn() as conn:
            db_rows = conn.execute(sql, params).fetchall()

        rows = _parse_rows(db_rows)

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


def _build_cluster(members: list[dict]) -> dict:
    """Build a cluster dict from its member rows."""
    # Medoid: member with highest avg cosine sim to all others (original space)
    vecs = np.array([m["embedding"] for m in members], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / np.where(norms == 0, 1, norms)
    sim_matrix = vecs_norm @ vecs_norm.T  # (k, k)
    avg_sims = sim_matrix.mean(axis=1)
    medoid_idx = int(avg_sims.argmax())

    # Date span
    times = [m["tweet_time"] or m["created_at"] for m in members]
    start_date = min(times)
    end_date = max(times)

    # Top mentioned users
    user_counts: dict[str, int] = {}
    for m in members:
        for u in (m["mentioned_users"] or []):
            user_counts[u] = user_counts.get(u, 0) + 1
    top_users = sorted(user_counts, key=user_counts.get, reverse=True)[:5]

    return {
        "medoid": members[medoid_idx],
        "count": len(members),
        "start_date": start_date,
        "end_date": end_date,
        "top_users": top_users,
        "members": members,
    }


def _cluster(rows: list[dict], max_topics: int = 10) -> list[dict]:
    """PCA + HDBSCAN clustering pipeline.

    Returns list of cluster dicts sorted by size descending.
    """
    n = len(rows)

    if not rows:
        return []

    if n < CLUSTER_MIN_SIZE:
        return [_build_cluster(rows)]

    # Extract embedding matrix and timestamps
    embeddings = np.array([r["embedding"] for r in rows], dtype=np.float32)
    timestamps = np.array([
        (r["tweet_time"] or r["created_at"]).timestamp()
        for r in rows
    ], dtype=np.float64)

    # PCA — cap components at n_rows
    n_components = min(n, PCA_N_COMPONENTS)
    reduced = PCA(n_components=n_components).fit_transform(embeddings)

    # Normalize timestamps to [0, 1] and scale
    t_min, t_max = timestamps.min(), timestamps.max()
    if t_max > t_min:
        t_norm = (timestamps - t_min) / (t_max - t_min)
    else:
        t_norm = np.zeros_like(timestamps)
    t_scaled = t_norm * TIME_WEIGHT

    # Append time as final dimension
    features = np.column_stack([reduced, t_scaled])

    # HDBSCAN
    min_samples = CLUSTER_MIN_SAMPLES if CLUSTER_MIN_SAMPLES is not None else CLUSTER_MIN_SIZE
    labels = HDBSCAN(
        min_cluster_size=CLUSTER_MIN_SIZE,
        min_samples=min_samples,
        copy=True,
    ).fit_predict(features)

    # Check if all noise
    unique_labels = set(labels)
    unique_labels.discard(-1)
    if not unique_labels:
        return [_build_cluster(rows)]

    # Build clusters
    clusters = []
    for label in unique_labels:
        members = [rows[i] for i in range(n) if labels[i] == label]
        clusters.append(_build_cluster(members))

    # Sort by size descending, truncate
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters[:max_topics]
