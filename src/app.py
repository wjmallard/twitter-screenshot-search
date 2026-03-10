"""Flask search GUI for screenshot search."""

import mimetypes
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file

import config
from db import (
    get_conn, search_fulltext, search_trigram, search_exact,
    count_fulltext, count_trigram, count_exact, count_screenshots,
)

app = Flask(__name__)

PER_PAGE = config.RESULTS_PER_PAGE


def _page_numbers(current, total):
    """Generate page numbers with ellipsis. Returns list of ints and None (for ellipsis)."""
    if total <= 5:
        return list(range(1, total + 1))
    # Build a window of 3 around current, clamped to edges
    win_start = max(1, min(current - 1, total - 3))
    win_end = min(total, max(current + 1, 4))
    pages = {1, total}
    for p in range(win_start, win_end + 1):
        pages.add(p)
    result = []
    for p in sorted(pages):
        if result and p - result[-1] > 1:
            result.append(None)
        result.append(p)
    return result


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    fuzzy = request.args.get("fuzzy", "word")
    sort = request.args.get("sort", "best")
    page = request.args.get("page", 1, type=int)
    offset = (page - 1) * PER_PAGE
    results = []
    total_results = 0

    if q:
        with get_conn() as conn:
            if fuzzy == "char":
                rows = search_trigram(conn, q, limit=PER_PAGE, offset=offset, sort=sort)
                total_results = count_trigram(conn, q)
            elif fuzzy == "none":
                rows = search_exact(conn, q, limit=PER_PAGE, offset=offset, sort=sort)
                total_results = count_exact(conn, q)
            else:
                rows = search_fulltext(conn, q, limit=PER_PAGE, offset=offset, sort=sort)
                total_results = count_fulltext(conn, q)
            for file_path, ocr_text, created_at, width, height, score in rows:
                results.append({
                    "file_path": file_path,
                    "name": Path(file_path).name,
                    "ocr_text": ocr_text or "",
                    "date": created_at.strftime("%Y-%m-%d · %I:%M %p") if created_at else "unknown",
                    "width": width,
                    "height": height,
                    "score": score,
                })

    total_pages = (total_results + PER_PAGE - 1) // PER_PAGE if total_results else 0

    with get_conn() as conn:
        total_indexed = count_screenshots(conn)

    return render_template(
        "index.html",
        q=q,
        fuzzy=fuzzy,
        sort=sort,
        page=page,
        total_pages=total_pages,
        results=results,
        total_indexed=total_indexed,
        total_results=total_results,
        pages=_page_numbers(page, total_pages),
    )


@app.route("/image")
def serve_image():
    path = request.args.get("path", "")
    if not path:
        abort(400)
    p = Path(path)
    if not p.is_file():
        abort(404)
    if ".." in p.parts:
        abort(403)

    if request.args.get("thumb"):
        from PIL import Image
        from io import BytesIO

        img = Image.open(p)
        img.thumbnail((800, 800))
        buf = BytesIO()
        fmt = "PNG" if p.suffix.lower() == ".png" else "JPEG"
        img.save(buf, format=fmt)
        buf.seek(0)
        mime = "image/png" if fmt == "PNG" else "image/jpeg"
        return send_file(buf, mimetype=mime)

    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    return send_file(p, mimetype=mime)


if __name__ == "__main__":
    app.run(debug=True, port=config.FLASK_PORT)
