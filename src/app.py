"""Flask search GUI for screenshot search."""

import mimetypes
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file

import config
from db import get_conn, search_fulltext, search_trigram, search_exact, count_screenshots

app = Flask(__name__)

PER_PAGE = 20


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    fuzzy = request.args.get("fuzzy", "word")
    sort = request.args.get("sort", "best")
    page = request.args.get("page", 1, type=int)
    offset = (page - 1) * PER_PAGE
    results = []

    if q:
        with get_conn() as conn:
            if fuzzy == "char":
                rows = search_trigram(conn, q, limit=PER_PAGE, offset=offset, sort=sort)
            elif fuzzy == "none":
                rows = search_exact(conn, q, limit=PER_PAGE, offset=offset, sort=sort)
            else:
                rows = search_fulltext(conn, q, limit=PER_PAGE, offset=offset, sort=sort)
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

    with get_conn() as conn:
        total_indexed = count_screenshots(conn)

    return render_template(
        "index.html",
        q=q,
        fuzzy=fuzzy,
        sort=sort,
        page=page,
        results=results,
        total_indexed=total_indexed,
        has_next=len(results) == PER_PAGE,
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
