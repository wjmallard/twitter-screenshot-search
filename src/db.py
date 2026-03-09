import psycopg
from contextlib import contextmanager

import config

DB_NAME = "twitter_screenshot_search"


@contextmanager
def get_conn():
    conn = psycopg.connect(dbname=DB_NAME)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_screenshot(conn, file_path, ocr_text, created_at, width, height):
    conn.execute(
        """
        INSERT INTO screenshots (file_path, ocr_text, created_at, width, height)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (file_path) DO UPDATE SET
            ocr_text = EXCLUDED.ocr_text,
            created_at = EXCLUDED.created_at,
            width = EXCLUDED.width,
            height = EXCLUDED.height
        """,
        (file_path, ocr_text, created_at, width, height),
    )


def images_in_db(conn) -> set[str]:
    rows = conn.execute("SELECT file_path FROM screenshots").fetchall()
    return {row[0] for row in rows}


_HALF_LIFE_SECS = config.DECAY_HALF_LIFE_DAYS * 86400
_DECAY = f"(1.0 / (EXTRACT(EPOCH FROM now() - COALESCE(created_at, now())) / {_HALF_LIFE_SECS} + 1))"

_FT_SCORE = "ts_rank(ocr_text_tsv, websearch_to_tsquery('english', %s))"
_TG_SCORE = "word_similarity(%s, ocr_text)"

SORT_OPTIONS = {
    "best": {"fulltext": f"{_FT_SCORE} * {_DECAY} DESC", "trigram": f"{_TG_SCORE} * {_DECAY} DESC"},
    "strongest": {"fulltext": f"{_FT_SCORE} DESC", "trigram": f"{_TG_SCORE} DESC"},
    "newest": {"fulltext": "created_at DESC NULLS LAST", "trigram": "created_at DESC NULLS LAST"},
    "oldest": {"fulltext": "created_at ASC NULLS LAST", "trigram": "created_at ASC NULLS LAST"},
}

# Number of extra %s params the ORDER BY clause needs for the query string
_SORT_EXTRA_PARAMS = {"best": 1, "strongest": 1, "newest": 0, "oldest": 0}


def search_fulltext(conn, query, limit=50, offset=0, sort="best"):
    if sort not in SORT_OPTIONS:
        sort = "best"
    order = SORT_OPTIONS[sort]["fulltext"]
    extra = _SORT_EXTRA_PARAMS[sort]
    params = (query, query) + (query,) * extra + (limit, offset)
    return conn.execute(
        f"""
        SELECT file_path, ocr_text, created_at, width, height,
               ts_rank(ocr_text_tsv, websearch_to_tsquery('english', %s)) AS score
        FROM screenshots
        WHERE ocr_text_tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY {order}
        LIMIT %s OFFSET %s
        """,
        params,
    ).fetchall()


def search_trigram(conn, query, limit=50, offset=0, sort="best"):
    if sort not in SORT_OPTIONS:
        sort = "best"
    order = SORT_OPTIONS[sort]["trigram"]
    extra = _SORT_EXTRA_PARAMS[sort]
    params = (query, query) + (query,) * extra + (limit, offset)
    return conn.execute(
        f"""
        SELECT file_path, ocr_text, created_at, width, height,
               word_similarity(%s, ocr_text) AS score
        FROM screenshots
        WHERE %s <<%% ocr_text
        ORDER BY {order}
        LIMIT %s OFFSET %s
        """,
        params,
    ).fetchall()


def count_screenshots(conn):
    row = conn.execute("SELECT count(*) FROM screenshots").fetchone()
    return row[0]
