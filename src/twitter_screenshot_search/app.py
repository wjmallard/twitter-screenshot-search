"""Flask search GUI for screenshot search."""

import mimetypes
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from twitter_screenshot_search import config
from twitter_screenshot_search.db import (
    get_conn, search_fulltext, search_trigram, search_exact,
    count_fulltext, count_trigram, count_exact, count_screenshots,
    load_all_signatures, get_screenshots_by_ids,
)
from twitter_screenshot_search.minhash import build_lsh_index, query_related

app = Flask(__name__)

PER_PAGE = config.RESULTS_PER_PAGE

# Build LSH index on startup
with get_conn() as conn:
    _sigs = load_all_signatures(conn)
print(f"Building LSH index from {len(_sigs)} signatures...")
_lsh, _minhashes = build_lsh_index(_sigs)
del _sigs
print("LSH index ready.")


def _format_size(size_bytes):
    """Format file size in human-readable units."""
    if size_bytes is None:
        return ""
    for unit in ("B", "kB", "MB", "GB"):
        if size_bytes < 1000 or unit == "GB":
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1000


def _page_numbers(current, total):
    """Generate page numbers with ellipsis. Always returns exactly 7 slots when total >= 7."""
    if total <= 7:
        return list(range(1, total + 1))
    # Near the start: 1 2 3 4 5 ... last
    if current <= 4:
        return [1, 2, 3, 4, 5, None, total]
    # Near the end: 1 ... n-4 n-3 n-2 n-1 n
    if current >= total - 3:
        return [1, None, total - 4, total - 3, total - 2, total - 1, total]
    # Middle: 1 ... c-1 c c+1 ... last
    return [1, None, current - 1, current, current + 1, None, total]


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
            for row_id, file_path, ocr_text, created_at_local, tz, width, height, file_size, score in rows:
                results.append({
                    "id": row_id,
                    "file_path": file_path,
                    "name": Path(file_path).name,
                    "ocr_text": ocr_text or "",
                    "date": created_at_local.strftime("%Y-%m-%d · %I:%M %p · %A") if created_at_local else "unknown",
                    "timezone": tz or "",
                    "width": width,
                    "height": height,
                    "file_size": _format_size(file_size),
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


def _format_screenshot(s, screenshot_id):
    return {
        "id": screenshot_id,
        "file_path": s["file_path"],
        "name": Path(s["file_path"]).name,
        "ocr_text": s["ocr_text"] or "",
        "date": s["created_at_local"].strftime("%Y-%m-%d · %I:%M %p · %A") if s["created_at_local"] else "unknown",
        "timezone": s["timezone"] or "",
        "width": s["width"],
        "height": s["height"],
        "file_size": _format_size(s["file_size"]),
    }


@app.route("/related/<int:screenshot_id>")
def related(screenshot_id):
    matches = query_related(_lsh, _minhashes, screenshot_id)
    match_ids = [mid for mid, _ in matches]
    sim_by_id = dict(matches)
    all_ids = [screenshot_id] + match_ids
    with get_conn() as conn:
        screenshots = get_screenshots_by_ids(conn, all_ids)
    source = None
    if screenshot_id in screenshots:
        source = _format_screenshot(screenshots[screenshot_id], screenshot_id)
    related_results = []
    for mid in match_ids:
        if mid not in screenshots:
            continue
        r = _format_screenshot(screenshots[mid], mid)
        r["similarity"] = round(sim_by_id[mid], 3)
        related_results.append(r)
    return jsonify({"source": source, "related": related_results})


def main():
    app.run(debug=True, port=config.FLASK_PORT)


if __name__ == "__main__":
    main()
