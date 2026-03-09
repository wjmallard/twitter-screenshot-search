import psycopg
from contextlib import contextmanager


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


SORT_OPTIONS = {
    "relevance": {"fulltext": "rank DESC", "trigram": "sim DESC"},
    "newest": {"fulltext": "created_at DESC NULLS LAST", "trigram": "created_at DESC NULLS LAST"},
    "oldest": {"fulltext": "created_at ASC NULLS LAST", "trigram": "created_at ASC NULLS LAST"},
}


def search_fulltext(conn, query, limit=50, offset=0, sort="relevance"):
    order = SORT_OPTIONS.get(sort, SORT_OPTIONS["relevance"])["fulltext"]
    return conn.execute(
        f"""
        SELECT file_path, ocr_text, created_at, width, height,
               ts_rank(ocr_text_tsv, websearch_to_tsquery('english', %s)) AS rank
        FROM screenshots
        WHERE ocr_text_tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY {order}
        LIMIT %s OFFSET %s
        """,
        (query, query, limit, offset),
    ).fetchall()


def search_trigram(conn, query, limit=50, offset=0, sort="relevance"):
    order = SORT_OPTIONS.get(sort, SORT_OPTIONS["relevance"])["trigram"]
    return conn.execute(
        f"""
        SELECT file_path, ocr_text, created_at, width, height,
               similarity(ocr_text, %s) AS sim
        FROM screenshots
        WHERE ocr_text %% %s
        ORDER BY {order}
        LIMIT %s OFFSET %s
        """,
        (query, query, limit, offset),
    ).fetchall()


def count_screenshots(conn):
    row = conn.execute("SELECT count(*) FROM screenshots").fetchone()
    return row[0]
