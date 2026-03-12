"""LM Studio embedding helpers and startup backfill."""

import logging
import sys

import httpx
from tqdm import tqdm

from ..db import get_conn
from .config import BACKFILL_BATCH_SIZE, EMBEDDING_MODEL, LMSTUDIO_URL

logging.getLogger("httpx").setLevel(logging.WARNING)


async def embed_texts(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Call LM Studio's OpenAI-compatible embeddings endpoint."""
    resp = await client.post(
        f"{LMSTUDIO_URL}/v1/embeddings",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [d["embedding"] for d in data]


def vec_literal(embedding: list[float]) -> str:
    """Format a float list as a pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def check_lmstudio():
    """Verify LM Studio is reachable and the embedding model is loaded."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{LMSTUDIO_URL}/v1/models", timeout=5.0)
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        print(
            f"Error: Cannot reach LM Studio at {LMSTUDIO_URL}\n"
            "Make sure LM Studio is running with the embedding model loaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    models = [m["id"] for m in resp.json().get("data", [])]
    if EMBEDDING_MODEL not in models:
        print(
            f"Error: Model '{EMBEDDING_MODEL}' not loaded in LM Studio.\n"
            f"Available models: {', '.join(models) or '(none)'}",
            file=sys.stderr,
        )
        sys.exit(1)


async def backfill_embeddings():
    """Embed all rows that have ocr_text but no embedding yet."""
    with get_conn() as conn:
        pending = conn.execute(
            "SELECT count(*) FROM screenshots "
            "WHERE ocr_text_clean IS NOT NULL AND ocr_text_clean != '' AND embedding IS NULL"
        ).fetchone()[0]

        if pending == 0:
            return

        progress = tqdm(total=pending, desc="Embedding backfill", file=sys.stderr)

        async with httpx.AsyncClient() as client:
            while True:
                rows = conn.execute(
                    "SELECT id, ocr_text_clean FROM screenshots "
                    "WHERE ocr_text_clean IS NOT NULL AND ocr_text_clean != '' AND embedding IS NULL "
                    "ORDER BY id LIMIT %s",
                    (BACKFILL_BATCH_SIZE,),
                ).fetchall()

                if not rows:
                    break

                ids = [r[0] for r in rows]
                texts = [r[1] for r in rows]

                try:
                    embeddings = await embed_texts(client, texts)
                except httpx.ConnectError:
                    progress.close()
                    print(
                        f"Cannot reach LM Studio at {LMSTUDIO_URL} — "
                        "backfill aborted, will retry next startup.",
                        file=sys.stderr,
                    )
                    return

                for row_id, emb in zip(ids, embeddings):
                    conn.execute(
                        "UPDATE screenshots SET embedding = %s::vector WHERE id = %s",
                        (vec_literal(emb), row_id),
                    )
                conn.commit()
                progress.update(len(ids))

        progress.close()
